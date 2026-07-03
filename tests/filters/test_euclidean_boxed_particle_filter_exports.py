from pyrecest import filters
from pyrecest.filters.euclidean_boxed_particle_filter import (
    BoxedParticleFilter,
    EuclideanBoxedParticleFilter,
)


def test_euclidean_boxed_particle_filter_is_lazy_exported():
    assert "EuclideanBoxedParticleFilter" in filters.__all__
    assert filters.EuclideanBoxedParticleFilter is EuclideanBoxedParticleFilter


def test_boxed_particle_filter_alias_is_lazy_exported():
    assert "BoxedParticleFilter" in filters.__all__
    assert filters.BoxedParticleFilter is BoxedParticleFilter
    assert filters.BoxedParticleFilter is filters.EuclideanBoxedParticleFilter
