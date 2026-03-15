from src.projection import create_projection_matrix, project


def test_projection_output_dim():
    proj = create_projection_matrix(input_dim=100, output_dim=16, seed=42)
    vec = [1.0] * 100
    result = project(vec, proj)
    assert len(result) == 16


def test_projection_deterministic():
    proj1 = create_projection_matrix(seed=42)
    proj2 = create_projection_matrix(seed=42)
    assert proj1[0][:5] == proj2[0][:5]
