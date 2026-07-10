from pathlib import Path


def replace_top_k_gate() -> None:
    path = Path("src/pyrecest/filters/association_hypotheses.py")
    lines = path.read_text(encoding="utf-8").splitlines()
    start_marker = "        accepted_keys = set()"
    end_marker = "        return result"
    print(f"source start marker count: {lines.count(start_marker)}")
    start = lines.index(start_marker)
    end = lines.index(end_marker, start) + 1
    replacement = [
        "        accepted_indices = set()",
        "        grouped: dict[int, list[tuple[int, AssociationHypothesis]]] = defaultdict(list)",
        "        for hypothesis_index, hypothesis in enumerate(hypotheses):",
        "            if hypothesis.is_missed_detection:",
        "                continue",
        "            key = (",
        "                _track_index(hypothesis)",
        "                if self.mode == \"track\"",
        "                else _measurement_index(hypothesis)",
        "            )",
        "            grouped[key].append((hypothesis_index, hypothesis))",
        "",
        "        for group in grouped.values():",
        "            sorted_group = sorted(",
        "                group,",
        "                key=lambda item: hypothesis_cost(",
        "                    item[1], missing_cost=self.missing_cost",
        "                ),",
        "            )",
        "            accepted_indices.update(",
        "                hypothesis_index",
        "                for hypothesis_index, _ in sorted_group[: self.k]",
        "            )",
        "",
        "        result = []",
        "        for hypothesis_index, hypothesis in enumerate(hypotheses):",
        "            if hypothesis.is_missed_detection:",
        "                result.append(hypothesis)",
        "                continue",
        "            accepted = hypothesis_index in accepted_indices",
        "            result.append(",
        "                hypothesis.with_acceptance(",
        "                    accepted,",
        "                    None if accepted else f\"top_{self.k}_{self.mode}_gate\",",
        "                )",
        "            )",
        "        return result",
    ]
    lines[start:end] = replacement
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_regression_test() -> None:
    path = Path("tests/filters/test_association_hypotheses.py")
    lines = path.read_text(encoding="utf-8").splitlines()
    test_marker = "    def test_top_k_gate_keeps_only_best_duplicate_pair_hypothesis(self):"
    anchor_marker = "    def test_top_k_gate_rejects_invalid_k(self):"
    print(f"test marker count: {lines.count(test_marker)}")
    print(f"test anchor count: {lines.count(anchor_marker)}")
    if test_marker in lines:
        raise RuntimeError("regression test already exists")
    anchor = lines.index(anchor_marker)
    addition = [
        "    def test_top_k_gate_keeps_only_best_duplicate_pair_hypothesis(self):",
        "        hypotheses = [",
        "            AssociationHypothesis(0, 0, cost=1.0),",
        "            AssociationHypothesis(0, 0, cost=100.0),",
        "            AssociationHypothesis(0, 1, cost=2.0),",
        "        ]",
        "",
        "        gated_with_rejections = filter_hypotheses(",
        "            hypotheses,",
        "            TopKGate(1, mode=\"track\"),",
        "            accepted_only=False,",
        "        )",
        "        self.assertEqual(",
        "            [hypothesis.accepted for hypothesis in gated_with_rejections],",
        "            [True, False, False],",
        "        )",
        "",
        "        gated = [",
        "            hypothesis",
        "            for hypothesis in gated_with_rejections",
        "            if hypothesis.accepted",
        "        ]",
        "        cost_matrix = hypotheses_to_cost_matrix(",
        "            gated,",
        "            num_tracks=1,",
        "            num_measurements=2,",
        "            missing_cost=99.0,",
        "        )",
        "        self.assertEqual(len(gated), 1)",
        "        self.assertEqual(gated[0].cost, 1.0)",
        "        self.assertAlmostEqual(cost_matrix[0, 0], 1.0)",
        "        self.assertEqual(cost_matrix[0, 1], 99.0)",
        "",
    ]
    lines[anchor:anchor] = addition
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    replace_top_k_gate()
    add_regression_test()
    Path(".github/workflows/apply-top-k-duplicate-fix.yml").unlink()
    Path(".github/apply_top_k_fix.py").unlink()
    print("TopKGate fix and regression test applied")


if __name__ == "__main__":
    main()
