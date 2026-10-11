import unittest
from analysis import checks, graph, FAMILIES


class CascadeChecks(unittest.TestCase):
    def test_hand_examples(self):
        checks()

    def test_exact_graph_sizes(self):
        for family in FAMILIES:
            g, _ = graph(family, 0)
            self.assertEqual(len(g), 200)
            self.assertEqual(g.number_of_edges(), 600)


if __name__ == "__main__":
    unittest.main()
