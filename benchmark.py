#!/usr/bin/env python3
"""
benchmark.py – Evaluate Recursive vs. Iterative Minimax Ghost agents.

Usage:
    python benchmark.py                    # Run 50 games per version (headless)
    python benchmark.py --display          # Run with visible game window
    python benchmark.py --test-equivalence # Verify both produce identical moves
    python benchmark.py --num-games 10     # Custom game count
"""

import sys
import time
import random
import argparse
import textwrap
import json

import pacman
import layout as layoutModule
import ghostAgents
import pacmanAgents
import multiAgents
import textDisplay
from game import Game


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_games(layoutObj, pacmanAgent, ghostCls, numGames, numGhosts, seed,
               show_display=False, frameTime=0.05):
    """Run *numGames* with the given ghost class and return per-game stats."""
    results = []
    rules = pacman.ClassicGameRules(timeout=300)

    for i in range(numGames):
        random.seed(seed + i)  # reproducible but different per game

        ghosts = [ghostCls(j + 1) for j in range(numGhosts)]
        if show_display:
            import graphicsDisplay
            display = graphicsDisplay.PacmanGraphics(zoom=1.0, frameTime=frameTime)
        else:
            display = textDisplay.NullGraphics()
        game = rules.newGame(layoutObj, pacmanAgent, ghosts, display,
                             quiet=True, catchExceptions=False)

        # Time each ghost decision
        total_decision_time = 0.0
        decision_count = 0
        max_metric = 0  # max recursion depth or max stack size

        original_getAction = ghostAgents.GhostAgent.getAction

        def timed_getAction(self_ghost, state, _orig=original_getAction):
            nonlocal total_decision_time, decision_count, max_metric
            t0 = time.perf_counter()
            action = _orig(self_ghost, state)
            t1 = time.perf_counter()
            total_decision_time += (t1 - t0)
            decision_count += 1
            if hasattr(self_ghost, '_maxRecursionDepth'):
                max_metric = max(max_metric, self_ghost._maxRecursionDepth)
            if hasattr(self_ghost, '_maxStackSize'):
                max_metric = max(max_metric, self_ghost._maxStackSize)
            return action

        # Monkey-patch for timing
        ghostAgents.GhostAgent.getAction = timed_getAction
        try:
            game.run()
        finally:
            ghostAgents.GhostAgent.getAction = original_getAction

        score = game.state.getScore()
        pacman_won = game.state.isWin()
        ghost_won = game.state.isLose()

        avg_time = (total_decision_time / decision_count) if decision_count else 0

        results.append({
            'score': score,
            'pacman_won': pacman_won,
            'ghost_won': ghost_won,
            'total_decision_time': total_decision_time,
            'avg_decision_time': avg_time,
            'decision_count': decision_count,
            'max_metric': max_metric,
        })
    return results


def _print_summary(label, results):
    n = len(results)
    ghost_wins = sum(1 for r in results if r['ghost_won'])
    pacman_wins = sum(1 for r in results if r['pacman_won'])
    avg_score = sum(r['score'] for r in results) / n
    avg_dt = sum(r['avg_decision_time'] for r in results) / n
    max_metric = max(r['max_metric'] for r in results)

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Games played          : {n}")
    print(f"  Ghost win rate        : {ghost_wins}/{n} ({100*ghost_wins/n:.1f}%)")
    print(f"  Pacman win rate       : {pacman_wins}/{n} ({100*pacman_wins/n:.1f}%)")
    print(f"  Avg Pacman score      : {avg_score:.1f}")
    print(f"  Avg decision time     : {avg_dt*1000:.3f} ms")
    print(f"  Max depth/stack size  : {max_metric}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Equivalence test
# ---------------------------------------------------------------------------

def test_equivalence(layoutName='minimaxClassic', numGhosts=1, depth=2, seed=42):
    """
    Run both ghost versions on the same layout with a fixed seed and verify
    they pick the same action at every step.
    """
    print(f"Testing move equivalence on '{layoutName}' (seed={seed}) ...")

    lay = layoutModule.getLayout(layoutName)
    if lay is None:
        print(f"ERROR: layout '{layoutName}' not found")
        return False

    random.seed(seed)
    pacAgent = multiAgents.ReflexAgent()

    numGhosts = min(numGhosts, lay.getNumGhosts())

    # Create both ghost types
    ghosts_rec = [ghostAgents.MinimaxGhostRecursive(j + 1, depth=depth)
                  for j in range(numGhosts)]
    ghosts_iter = [ghostAgents.MinimaxGhostIterative(j + 1, depth=depth)
                   for j in range(numGhosts)]

    rules = pacman.ClassicGameRules(timeout=300)
    display = textDisplay.NullGraphics()

    # Build two identical initial states
    random.seed(seed)
    game_rec = rules.newGame(lay, pacAgent, ghosts_rec, display, quiet=True)
    random.seed(seed)
    game_iter = rules.newGame(lay, pacAgent, ghosts_iter, display, quiet=True)

    state_rec = game_rec.state
    state_iter = game_iter.state

    step = 0
    mismatches = 0
    numAgents = state_rec.getNumAgents()

    while not (state_rec.isWin() or state_rec.isLose()):
        for agentIdx in range(numAgents):
            if state_rec.isWin() or state_rec.isLose():
                break

            if agentIdx == 0:
                # Pacman: use the same agent for both
                action = pacAgent.getAction(state_rec)
                state_rec = state_rec.generateSuccessor(0, action)
                state_iter = state_iter.generateSuccessor(0, action)
            else:
                # Ghost
                if agentIdx <= numGhosts:
                    dist_rec = ghosts_rec[agentIdx - 1].getDistribution(state_rec)
                    dist_iter = ghosts_iter[agentIdx - 1].getDistribution(state_iter)

                    action_rec = max(dist_rec, key=dist_rec.get)
                    action_iter = max(dist_iter, key=dist_iter.get)

                    if action_rec != action_iter:
                        mismatches += 1
                        print(f"  MISMATCH at step {step}, ghost {agentIdx}: "
                              f"recursive={action_rec}, iterative={action_iter}")

                    state_rec = state_rec.generateSuccessor(agentIdx, action_rec)
                    state_iter = state_iter.generateSuccessor(agentIdx, action_rec)

        step += 1
        if step > 500:
            print("  (stopping after 500 rounds)")
            break

    if mismatches == 0:
        print(f"  ✓ PASS – {step} rounds, all ghost moves identical.")
        return True
    else:
        print(f"  ✗ FAIL – {mismatches} mismatches in {step} rounds.")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark recursive vs. iterative minimax ghosts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python benchmark.py
              python benchmark.py --test-equivalence
              python benchmark.py --num-games 10 --layout smallClassic
        """))
    parser.add_argument('--test-equivalence', action='store_true',
                        help='Verify both versions produce identical moves')
    parser.add_argument('--num-games', type=int, default=50,
                        help='Number of games per version (default: 50)')
    parser.add_argument('--layout', default='smallClassic',
                        help='Layout to use (default: smallClassic)')
    parser.add_argument('--num-ghosts', type=int, default=2,
                        help='Number of ghosts (default: 2)')
    parser.add_argument('--depth', type=int, default=2,
                        help='Minimax search depth (default: 2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')
    parser.add_argument('--display', action='store_true',
                        help='Show the Pacman GUI during games (default: headless)')
    parser.add_argument('--frame-time', type=float, default=0.05,
                        help='Seconds between frames when --display is on (default: 0.05)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Save results to a JSON file (e.g. --output results.json)')

    args = parser.parse_args()

    if args.test_equivalence:
        ok = test_equivalence(layoutName=args.layout,
                              numGhosts=args.num_ghosts,
                              depth=args.depth,
                              seed=args.seed)
        sys.exit(0 if ok else 1)

    lay = layoutModule.getLayout(args.layout)
    if lay is None:
        print(f"ERROR: layout '{args.layout}' not found")
        sys.exit(1)

    numGhosts = min(args.num_ghosts, lay.getNumGhosts())
    pacAgent = multiAgents.ReflexAgent()

    print(f"Layout: {args.layout}  |  Ghosts: {numGhosts}  |  "
          f"Depth: {args.depth}  |  Games per version: {args.num_games}")

    # --- Version A: Recursive ---
    print(f"\nRunning Version A (MinimaxGhostRecursive) × {args.num_games} games ...")
    results_rec = _run_games(
        lay, pacAgent,
        lambda idx: ghostAgents.MinimaxGhostRecursive(idx, depth=args.depth),
        args.num_games, numGhosts, args.seed,
        show_display=args.display, frameTime=args.frame_time)
    _print_summary("Version A – Recursive Minimax (alpha-beta)", results_rec)

    # --- Version B: Iterative ---
    print(f"Running Version B (MinimaxGhostIterative) × {args.num_games} games ...")
    results_iter = _run_games(
        lay, pacAgent,
        lambda idx: ghostAgents.MinimaxGhostIterative(idx, depth=args.depth),
        args.num_games, numGhosts, args.seed,
        show_display=args.display, frameTime=args.frame_time)
    _print_summary("Version B – Iterative Minimax (stack frames)", results_iter)

    # --- Comparison table ---
    print("=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    n = args.num_games
    gw_a = sum(1 for r in results_rec if r['ghost_won'])
    gw_b = sum(1 for r in results_iter if r['ghost_won'])
    avg_a = sum(r['avg_decision_time'] for r in results_rec) / n * 1000
    avg_b = sum(r['avg_decision_time'] for r in results_iter) / n * 1000
    print(f"  {'Metric':<25} {'Recursive':>12} {'Iterative':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Ghost Win Rate':<25} {f'{gw_a}/{n}':>12} {f'{gw_b}/{n}':>12}")
    print(f"  {'Avg Decision Time (ms)':<25} {avg_a:>12.3f} {avg_b:>12.3f}")
    print(f"  {'Max Depth/Stack':<25} "
          f"{max(r['max_metric'] for r in results_rec):>12} "
          f"{max(r['max_metric'] for r in results_iter):>12}")
    print("=" * 60)

    # --- Save to file if requested ---
    if args.output:
        pw_a = sum(1 for r in results_rec if r['pacman_won'])
        pw_b = sum(1 for r in results_iter if r['pacman_won'])
        output_data = {
            'config': {
                'layout': args.layout,
                'num_ghosts': numGhosts,
                'depth': args.depth,
                'num_games': args.num_games,
                'seed': args.seed,
            },
            'version_a_recursive': {
                'games': results_rec,
                'ghost_win_rate': f'{gw_a}/{n}',
                'ghost_win_pct': round(100 * gw_a / n, 1),
                'pacman_win_rate': f'{pw_a}/{n}',
                'pacman_win_pct': round(100 * pw_a / n, 1),
                'avg_score': round(sum(r['score'] for r in results_rec) / n, 1),
                'avg_decision_time_ms': round(avg_a, 3),
                'max_recursion_depth': max(r['max_metric'] for r in results_rec),
            },
            'version_b_iterative': {
                'games': results_iter,
                'ghost_win_rate': f'{gw_b}/{n}',
                'ghost_win_pct': round(100 * gw_b / n, 1),
                'pacman_win_rate': f'{pw_b}/{n}',
                'pacman_win_pct': round(100 * pw_b / n, 1),
                'avg_score': round(sum(r['score'] for r in results_iter) / n, 1),
                'avg_decision_time_ms': round(avg_b, 3),
                'max_stack_size': max(r['max_metric'] for r in results_iter),
            },
            'comparison_summary': {
                'ghost_win_rate': {
                    'recursive': f'{gw_a}/{n} ({round(100*gw_a/n,1)}%)',
                    'iterative': f'{gw_b}/{n} ({round(100*gw_b/n,1)}%)',
                },
                'pacman_win_rate': {
                    'recursive': f'{pw_a}/{n} ({round(100*pw_a/n,1)}%)',
                    'iterative': f'{pw_b}/{n} ({round(100*pw_b/n,1)}%)',
                },
                'avg_decision_time_ms': {
                    'recursive': round(avg_a, 3),
                    'iterative': round(avg_b, 3),
                },
                'max_depth_or_stack': {
                    'recursive': max(r['max_metric'] for r in results_rec),
                    'iterative': max(r['max_metric'] for r in results_iter),
                },
                'avg_score': {
                    'recursive': round(sum(r['score'] for r in results_rec) / n, 1),
                    'iterative': round(sum(r['score'] for r in results_iter) / n, 1),
                },
            },
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n  Results saved to: {args.output}")


if __name__ == '__main__':
    main()
