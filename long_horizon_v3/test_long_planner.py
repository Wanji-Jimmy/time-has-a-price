"""Artificial kernels only: these tests never inspect market outcomes.

The long planner uses the old, audited financial transition rules.  Tests here
check the new Bellman recombination against independent scalar arithmetic and
the old exact short-tree solver, rather than mirroring its implementation.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from long_horizon_v3.long_planner import FullHorizonPlanner, PlannerConfig
from time_budget_policy_v2.planner import choose as exact_short
from time_budget_policy_v2.planner import settle_retirement, transition, utility


def flat_kernel(body=1., gap=1.):
    return {"probabilities": np.ones((1, 1)), "gaps": np.array([[gap]]),
            "intraday": np.array([[body]]), "next_state": np.zeros((1, 1), int)}


def delayed_kernel(final_body=2.):
    return {"probabilities": np.ones((4, 1)), "gaps": np.ones((4, 1)),
            "intraday": np.array([[1.], [1.], [1.], [final_body]]),
            "next_state": np.array([[1], [2], [3], [3]])}


class LongHorizonTests(unittest.TestCase):
    def cfg(self, **changes):
        return replace(PlannerConfig(), commission=0., slippage=0., **changes)

    def test_one_step_matches_exact_all_q_including_close_settlement(self):
        cfg = self.cfg(horizon_steps=1)
        ker = {"probabilities": np.array([[.3, .7]]), "gaps": np.array([[.8, 1.2]]),
               "intraday": np.array([[.9, 1.15]]), "next_state": np.zeros((1, 2), int)}
        model = FullHorizonPlanner(ker, .1, config=cfg).build()
        for b, h in ((100., 0.), (50., 50.), (0., 100.)):
            expected = exact_short(b, h, 90., 0, 1, 0, .1, ker, depth=1, config=cfg)
            actual = model.choose(b, h, 90., 0, 1, 0)
            np.testing.assert_allclose(actual["q_values"], expected["q_values"], atol=1e-12, rtol=0)
            self.assertEqual(int(actual["action_index"]), int(expected["action_index"]))

    def test_two_step_matches_exact_contingent_tree_on_grid(self):
        cfg = self.cfg(horizon_steps=2)
        ker = {"probabilities": np.array([[.5, .5], [1., 0.], [1., 0.]]),
               "gaps": np.ones((3, 2)),
               "intraday": np.array([[1., 1.], [1.2, 1.2], [.8, .8]]),
               "next_state": np.array([[1, 2], [1, 1], [2, 2]])}
        model = FullHorizonPlanner(ker, .02, config=cfg).build()
        expected = exact_short(100., 0., 90., 0, 2, 0, .02, ker, depth=2, config=cfg)
        actual = model.choose(100., 0., 90., 0, 2, 0)
        np.testing.assert_allclose(actual["q_values"], expected["q_values"], atol=1e-12, rtol=0)
        # The first flat session buys information.  Future bad-state branches
        # retire, while good-state branches take the rising market.
        self.assertEqual(int(actual["action_index"]), 1)

    def test_four_steps_find_opportunity_hidden_from_two_steps(self):
        cfg = self.cfg(horizon_steps=4, initial_life=5.)
        ker = delayed_kernel()
        short = exact_short(100., 0., 5., 0, 4, 0, 0., ker, depth=2, config=cfg)
        model = FullHorizonPlanner(ker, 0., config=cfg).build()
        long = model.choose(100., 0., 5., 0, 4, 0)
        self.assertEqual(int(short["action_index"]), 0)
        self.assertEqual(int(long["action_index"]), 4)
        self.assertAlmostEqual(float(short["planned_utility"]), math.log(2.))
        self.assertAlmostEqual(float(long["planned_utility"]), math.log(2.68), places=12)
        # Replay actual selected actions using the independent old transition.
        b, h, life, packs, market = 100., 0., 5., 0, 0
        for r in range(4, 0, -1):
            action = int(model.choose(b, h, life, packs, r, market)["action_index"])
            self.assertNotEqual(action, 0)
            out = transition(b, h, life, packs, r, action, 1.,
                             float(ker["intraday"][market, 0]), int(ker["next_state"][market, 0]),
                             0., cfg)
            b, h, life, packs, market = (float(out["cash"]), float(out["holding"]),
                float(out["life"]), int(out["packs_used"]), int(out["market_state"]))
        self.assertEqual(packs, 2)
        self.assertAlmostEqual(b, 168.)
        self.assertAlmostEqual(life, 15.)

    def test_longer_horizon_is_not_an_instruction_to_buy_time(self):
        cfg = self.cfg(horizon_steps=4, initial_life=5.)
        model = FullHorizonPlanner(delayed_kernel(.5), 0., config=cfg).build()
        actual = model.choose(100., 0., 5., 0, 4, 0)
        self.assertEqual(int(actual["action_index"]), 0)
        self.assertAlmostEqual(float(actual["planned_utility"]), math.log(2.))

    def test_death_precedes_true_horizon_terminal_conversion(self):
        cfg = self.cfg(horizon_steps=1, initial_life=5.)
        model = FullHorizonPlanner(flat_kernel(1.1), .1, config=cfg).build()
        q = model.choose(100., 0., 5., 0, 1, 0)["q_values"]
        np.testing.assert_array_equal(q[1:4], np.zeros(3))
        self.assertTrue(np.all(q[4:] > 0))
        # Buying after physical death must not resurrect the account.
        dead_cfg = self.cfg(horizon_steps=2, initial_life=5.)
        dead = FullHorizonPlanner(flat_kernel(), .1, config=dead_cfg).build()
        np.testing.assert_array_equal(dead.choose(100., 0., 0., 0, 1, 0)["q_values"], np.zeros(7))

    def test_unknown_opening_gap_uses_expected_utility_not_mean_wealth(self):
        cfg = self.cfg(horizon_steps=1)
        ker = {"probabilities": np.array([[.5, .5]]), "gaps": np.array([[.5, 1.5]]),
               "intraday": np.ones((1, 2)), "next_state": np.zeros((1, 2), int)}
        actual = FullHorizonPlanner(ker, 0., config=cfg).build().choose(0., 100., 90., 0, 1, 0)
        expected_retire = .5 * math.log(1.5) + .5 * math.log(2.5)
        self.assertAlmostEqual(float(actual["q_values"][0]), expected_retire, places=12)
        self.assertLess(float(actual["q_values"][0]), math.log(2.))
        # An all-cash entry occurs at the open, so it does not capture that
        # unknown gap.  The true last step settles its flat intraday at close.
        cash = FullHorizonPlanner(ker, 0., config=cfg).build().choose(100., 0., 90., 0, 1, 0)
        self.assertAlmostEqual(float(cash["q_values"][3]), math.log(2.), places=12)

    def test_rejected_pack_and_zero_probability_padding(self):
        cfg = self.cfg(horizon_steps=1, initial_life=5.)
        risky = {"probabilities": np.array([[.5, .5]]), "gaps": np.array([[.1, 1.]]),
                 "intraday": np.ones((1, 2)), "next_state": np.zeros((1, 2), int)}
        q = FullHorizonPlanner(risky, 0., config=cfg).build().choose(0., 20., 5., 0, 1, 0)["q_values"]
        self.assertTrue(np.all(np.isneginf(q[4:])))
        padding = dict(risky, probabilities=np.array([[0., 1.]]), intraday=np.array([[1., 1.5]]))
        safe = FullHorizonPlanner(padding, 0., config=cfg).build().choose(0., 100., 5., 0, 1, 0)
        self.assertTrue(np.all(np.isfinite(safe["q_values"])))
        self.assertEqual(int(safe["action_index"]), 6)

    def test_terminal_only_ablation_still_allows_same_consumption_purchases(self):
        cfg = self.cfg(horizon_steps=2)
        model = FullHorizonPlanner(flat_kernel(), .5, config=cfg, allow_mid_buy=False).build()
        out = model.choose(100., 0., 90., 0, 2, 0)
        self.assertTrue(np.all(np.isneginf(out["q_values"][4:])))
        self.assertEqual(int(out["action_index"]), 0)
        terminal = settle_retirement(100., 0., 90., 0, .5, cfg)
        self.assertEqual(int(terminal["packs_bought"]), 6)
        self.assertAlmostEqual(float(out["planned_utility"]), float(terminal["utility"]))

    def test_terminal_only_ablation_disables_future_not_only_root_purchases(self):
        cfg = self.cfg(horizon_steps=4, initial_life=20.)
        allowed = FullHorizonPlanner(delayed_kernel(), 0., config=cfg).build()
        forbidden = FullHorizonPlanner(delayed_kernel(), 0., config=cfg, allow_mid_buy=False).build()
        yes = allowed.choose(100., 0., 20., 0, 4, 0)
        no = forbidden.choose(100., 0., 20., 0, 4, 0)
        # Initial life suffices for the first three flat sessions, but the
        # fourth profitable session requires a later time purchase.
        self.assertEqual(int(yes["action_index"]), 1)
        self.assertAlmostEqual(float(yes["planned_utility"]), math.log(2.84), places=12)
        self.assertEqual(int(no["action_index"]), 0)
        self.assertAlmostEqual(float(no["planned_utility"]), math.log(2.), places=12)

    def test_free_time_is_capped_and_alpha_zero_ties_buy_no_terminal_pack(self):
        cfg = self.cfg(horizon_steps=1, pack_cost=0.)
        for alpha, expected_packs in ((.1, 6), (0., 0)):
            out = settle_retirement(100., 0., 90., 0, alpha, cfg)
            self.assertEqual(int(out["packs_bought"]), expected_packs)
            actual = FullHorizonPlanner(flat_kernel(), alpha, config=cfg).build().choose(100., 0., 90., 0, 1, 0)
            self.assertEqual(int(actual["action_index"]), 0)
            self.assertAlmostEqual(float(actual["planned_utility"]), float(out["utility"]))

    def test_life_invariant_depth_and_out_of_domain_queries_are_explicit(self):
        cfg = self.cfg(horizon_steps=3)
        model = FullHorizonPlanner(flat_kernel(), .1, config=cfg).build()
        with self.assertRaisesRegex(ValueError, "life disagrees"):
            model.choose(100., 0., 89., 0, 3, 0)
        with self.assertRaisesRegex(ValueError, "depth"):
            model.choose(100., 0., 90., 0, 3, 0, depth=2)
        with self.assertRaisesRegex(ValueError, "wealth"):
            model.choose(121., 0., 90., 0, 3, 0)
        with self.assertRaisesRegex(ValueError, "different alpha"):
            model.choose(100., 0., 90., 0, 3, 0, alpha=.5)
        self.assertEqual(model.metadata()["maximum_internal_wealth_extrapolation"], 0.)

    def test_broadcast_and_single_state_results_agree(self):
        cfg = self.cfg(horizon_steps=2)
        model = FullHorizonPlanner(flat_kernel(1.01), .1, config=cfg).build()
        batch = model.choose([100., 50.], [0., 50.], 90., 0, 2, 0)
        for i, pair in enumerate(((100., 0.), (50., 50.))):
            single = model.choose(*pair, 90., 0, 2, 0)
            np.testing.assert_array_equal(batch["q_values"][i], single["q_values"])


if __name__ == "__main__":
    unittest.main()
