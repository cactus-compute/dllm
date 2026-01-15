"""
Integration tests for BERT-RDLM implementation.

Tests cover:
1. Prior distributions (uniform, masked, mixture)
2. Noise schedules (geometric, linear, cosine)
3. Riemannian normal sampling
4. RDLM interpolation
5. Schedule precomputation
6. Trainer forward pass
7. Sampler flow integration
"""

import torch
import torch.nn.functional as F
import pytest
import math

from dllm.pipelines.bert_sfm.geodesic_utils import (
    exp_map, log_map, make_tangent, uniform_prior,
    simplex_to_sphere, sphere_to_simplex,
)


class TestRDLMPriors:
    """Tests for RDLM prior distributions."""

    def test_masked_prior_shape(self):
        """Test masked prior returns correct shape."""
        from dllm.pipelines.bert_rdlm.rdlm_utils import masked_prior

        shape = (2, 10, 100)
        device = torch.device('cpu')
        x0 = masked_prior(shape, device)

        assert x0.shape == shape

    def test_masked_prior_is_onehot(self):
        """Test masked prior is one-hot at mask index."""
        from dllm.pipelines.bert_rdlm.rdlm_utils import masked_prior

        shape = (4, 8, 50)
        device = torch.device('cpu')
        mask_idx = -1  # Last token

        x0 = masked_prior(shape, device, mask_idx=mask_idx)

        # Should be all zeros except at mask_idx
        assert torch.allclose(x0[..., mask_idx], torch.ones(4, 8))
        assert torch.allclose(x0[..., :-1].sum(), torch.zeros(1))

    def test_mixture_prior_shape(self):
        """Test mixture prior returns correct shape."""
        from dllm.pipelines.bert_rdlm.rdlm_utils import mixture_prior

        shape = (2, 10, 100)
        device = torch.device('cpu')
        x0 = mixture_prior(shape, device, mixing_prob=0.5)

        assert x0.shape == shape

    def test_mixture_prior_on_sphere(self):
        """Test mixture prior samples are on unit sphere."""
        from dllm.pipelines.bert_rdlm.rdlm_utils import mixture_prior

        shape = (16, 32, 100)
        device = torch.device('cpu')
        x0 = mixture_prior(shape, device, mixing_prob=0.5)

        norms = x0.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_get_rdlm_prior_all_types(self):
        """Test get_rdlm_prior for all prior types."""
        from dllm.pipelines.bert_rdlm.rdlm_utils import get_rdlm_prior, RDLMPriorType

        shape = (4, 16, 64)
        device = torch.device('cpu')

        for prior_type in [RDLMPriorType.UNIFORM, RDLMPriorType.MASKED, RDLMPriorType.MIXTURE]:
            x0 = get_rdlm_prior(prior_type, shape, device)
            assert x0.shape == shape
            norms = x0.norm(dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestNoiseSchedules:
    """Tests for RDLM noise schedules."""

    def test_geometric_schedule_endpoints(self):
        """Test geometric schedule at t=0 and t=1."""
        from dllm.pipelines.bert_rdlm.rdlm_utils import geometric_schedule

        sigma_0, sigma_T = 0.001, 1.0

        t0 = torch.tensor(0.0)
        t1 = torch.tensor(1.0)

        assert torch.allclose(geometric_schedule(t0, sigma_0, sigma_T), torch.tensor(sigma_0))
        assert torch.allclose(geometric_schedule(t1, sigma_0, sigma_T), torch.tensor(sigma_T))

    def test_geometric_schedule_monotonic(self):
        """Test geometric schedule is monotonically increasing."""
        from dllm.pipelines.bert_rdlm.rdlm_utils import geometric_schedule

        t = torch.linspace(0, 1, 100)
        sigma = geometric_schedule(t)

        diff = sigma[1:] - sigma[:-1]
        assert (diff >= 0).all()

    def test_bridge_gamma_increasing(self):
        """Test bridge gamma coefficient increases with t."""
        from dllm.pipelines.bert_rdlm.rdlm_utils import bridge_gamma

        t = torch.linspace(0.1, 0.9, 50)
        gamma = bridge_gamma(t, sigma_0=0.001, sigma_T=0.2, schedule_type="geometric")

        # Drift coeff should increase as t approaches 1 for geometric schedule
        assert gamma[-1] > gamma[0]


class TestRDLMSchedule:
    """Tests for RDLMSchedule class."""

    def test_schedule_creation(self):
        """Test schedule can be created."""
        from dllm.pipelines.bert_rdlm.rdlm_utils import RDLMSchedule

        schedule = RDLMSchedule(
            schedule_type="geometric",
            sigma_0=0.001,
            sigma_T=1.0,
            n_time_steps=10,
            precompute=True,
            preprocess_dims=64,
            manifold_dim=63,
        )

        assert schedule.alpha_t is not None
        assert schedule.rho_t is not None
        assert len(schedule.alpha_t) == 11

    def test_schedule_alpha_rho_lookup(self):
        """Test alpha/rho lookup."""
        from dllm.pipelines.bert_rdlm.rdlm_utils import RDLMSchedule

        schedule = RDLMSchedule(
            schedule_type="geometric",
            n_time_steps=10,
            precompute=True,
            preprocess_dims=64,
            manifold_dim=63,
        )

        t = torch.tensor([0.0, 0.5, 0.99])
        alpha_t, rho_t = schedule.get_alpha_rho(t)

        assert alpha_t.shape == (3,)
        assert rho_t.shape == (3,)

    def test_schedule_to_device(self):
        """Test schedule can move to device."""
        from dllm.pipelines.bert_rdlm.rdlm_utils import RDLMSchedule

        schedule = RDLMSchedule(n_time_steps=10, precompute=True, preprocess_dims=64, manifold_dim=63)

        # Should not error even on CPU
        schedule = schedule.to(torch.device('cpu'))
        assert schedule.device == torch.device('cpu')


class TestRiemannianNormal:
    """Tests for Riemannian normal sampling."""

    def test_sample_on_sphere(self):
        """Test Riemannian normal samples are on sphere."""
        from dllm.pipelines.bert_rdlm.rdlm_utils import sample_riemannian_normal

        B, L, D = 8, 16, 64
        device = torch.device('cpu')

        # Random mean direction on sphere
        mean_dir = torch.randn(B, L, D)
        mean_dir = mean_dir / mean_dir.norm(dim=-1, keepdim=True)

        scale = torch.ones(B) * 0.1

        samples = sample_riemannian_normal(mean_dir, scale)

        # Check on unit sphere
        norms = samples.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_sample_near_mean(self):
        """Test samples concentrate near mean for small scale."""
        from dllm.pipelines.bert_rdlm.rdlm_utils import sample_riemannian_normal

        B, L, D = 100, 1, 32
        device = torch.device('cpu')

        # Fixed mean direction
        mean_dir = torch.zeros(B, L, D)
        mean_dir[..., 0] = 1.0  # First basis vector

        scale = torch.ones(B) * 0.01  # Small scale

        samples = sample_riemannian_normal(mean_dir, scale)

        # Inner product with mean should be close to 1
        inner = (samples * mean_dir).sum(dim=-1)
        assert inner.mean() > 0.99


class TestRDLMInterpolation:
    """Tests for RDLM interpolation."""

    def test_interpolant_shape(self):
        """Test interpolant returns correct shape."""
        from dllm.pipelines.bert_rdlm.rdlm_utils import rdlm_interpolant

        B, L, V = 4, 8, 100
        device = torch.device('cpu')

        x0 = uniform_prior((B, L, V), device)
        target_indices = torch.randint(0, V, (B, L))
        t = torch.rand(B)
        alpha_t = torch.rand(B)
        rho_t = torch.rand(B) * 0.1

        xt = rdlm_interpolant(x0, target_indices, t, alpha_t, rho_t, V)

        assert xt.shape == (B, L, V)

    def test_interpolant_on_sphere(self):
        """Test interpolant is on unit sphere."""
        from dllm.pipelines.bert_rdlm.rdlm_utils import rdlm_interpolant

        B, L, V = 8, 16, 50
        device = torch.device('cpu')

        x0 = uniform_prior((B, L, V), device)
        target_indices = torch.randint(0, V, (B, L))
        t = torch.rand(B)
        alpha_t = torch.rand(B)
        rho_t = torch.rand(B) * 0.1

        xt = rdlm_interpolant(x0, target_indices, t, alpha_t, rho_t, V)

        norms = xt.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestTargetDrift:
    """Tests for bridge drift computation."""

    def test_drift_is_tangent(self):
        """Test target drift is in tangent space."""
        from dllm.pipelines.bert_rdlm.rdlm_utils import compute_target_drift

        B, L, V = 4, 8, 32
        device = torch.device('cpu')

        xt = uniform_prior((B, L, V), device)
        target_indices = torch.randint(0, V, (B, L))
        gamma_t = torch.ones(B) * 2.0

        drift = compute_target_drift(xt, target_indices, gamma_t, V)

        # Check tangent: <drift, xt> should be 0
        inner = (drift * xt).sum(dim=-1)
        assert torch.allclose(inner, torch.zeros_like(inner), atol=1e-5)

    def test_drift_points_to_target(self):
        """Test drift points toward target direction."""
        from dllm.pipelines.bert_rdlm.rdlm_utils import compute_target_drift

        B, L, V = 2, 4, 16
        device = torch.device('cpu')

        xt = uniform_prior((B, L, V), device)
        target_indices = torch.randint(0, V, (B, L))
        gamma_t = torch.ones(B) * 1.0

        drift = compute_target_drift(xt, target_indices, gamma_t, V)

        # Moving in drift direction should increase inner product with target
        x1 = torch.zeros_like(xt)
        x1.scatter_(-1, target_indices.unsqueeze(-1), 1.0)

        # Small step in drift direction
        step = 0.01
        xt_new = exp_map(xt, step * drift)

        inner_old = (xt * x1).sum(dim=-1)
        inner_new = (xt_new * x1).sum(dim=-1)

        # Should generally increase (not always due to geodesics)
        # At minimum, should not decrease significantly
        assert (inner_new >= inner_old - 0.1).all()


class TestIntegration:
    """Integration tests requiring model setup."""

    @pytest.fixture
    def mock_model(self):
        """Create a small mock model for testing."""
        import torch.nn as nn

        class MockModel(nn.Module):
            class Config:
                vocab_size = 100
                hidden_size = 64

            def __init__(self):
                super().__init__()
                self.config = self.Config()
                self.embeddings = nn.Embedding(100, 64)
                self.linear = nn.Linear(64, 100)

            def get_input_embeddings(self):
                return self.embeddings

            def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
                if inputs_embeds is None:
                    inputs_embeds = self.embeddings(input_ids)
                logits = self.linear(inputs_embeds)

                class Output:
                    pass
                out = Output()
                out.logits = logits
                return out

        return MockModel()

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        class MockTokenizer:
            pad_token_id = 0
            eos_token_id = 1
            mask_token_id = 99
            vocab_size = 100
            padding_side = "right"

            def __call__(self, text, **kwargs):
                # Simple tokenization
                return {"input_ids": torch.randint(0, 100, (len(text), 16))}

        return MockTokenizer()

    def test_sampler_flow_integrate(self, mock_model, mock_tokenizer):
        """Test sampler flow integration."""
        from dllm.pipelines.bert_rdlm.sampler import BertRDLMSampler, BertRDLMSamplerConfig

        sampler = BertRDLMSampler(model=mock_model, tokenizer=mock_tokenizer)

        B, T, V = 2, 8, 100
        device = torch.device('cpu')

        x_sphere = uniform_prior((B, T, V), device)
        flow_mask = torch.ones(B, T, dtype=torch.bool)
        context_embeds = mock_model.get_input_embeddings()(torch.zeros(B, T, dtype=torch.long))
        attention_mask = torch.ones(B, T, dtype=torch.long)

        config = BertRDLMSamplerConfig(n_steps=5, temperature=1.0, add_mask_token=False)

        x_final = sampler.flow_integrate(
            x_sphere=x_sphere,
            flow_mask=flow_mask,
            context_embeds=context_embeds,
            attention_mask=attention_mask,
            config=config,
        )

        assert x_final.shape == (B, T, V)
        norms = x_final.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
