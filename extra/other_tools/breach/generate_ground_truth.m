function generate_ground_truth()
%GENERATE_GROUND_TRUTH  Compute Breach STL robustness for tidystl test cases.
%
%   Requires Breach on the MATLAB path. Run InitBreach first if needed.
%
%   Usage:
%     >> cd /path/to/tidystl
%     >> generate_ground_truth
%
%   Writes CSV files to tests/breach_ground_truth/

    out = fullfile(pwd, 'tests', 'breach_ground_truth');
    if ~exist(out, 'dir')
        mkdir(out);
    end

    fprintf('Generating Breach ground truth -> %s\n\n', out);
    n = 0;

    %% ======== Basic operators ========

    % 1. Simple predicate: x >= 3
    n = n + eval_1sig(out, 'predicate_gte', ...
        [0 1 2 3 4], [5 2 4 1 6], ...
        'x[t] - 3 > 0');

    % 2. Negation: not (x >= 0)
    n = n + eval_1sig(out, 'not_simple', ...
        [0 1 2 3 4], [3 -1 2 -4 5], ...
        'not (x[t] > 0)');

    % 3. Conjunction: (x >= 0) and (y >= 0)
    n = n + eval_2sig(out, 'and_two', ...
        [0 1 2 3 4], [3 -1 2 4 -2], [1 2 -1 3 5], ...
        '(x[t] > 0) and (y[t] > 0)');

    % 4. Disjunction: (x >= 0) or (y >= 0)
    n = n + eval_2sig(out, 'or_two', ...
        [0 1 2 3 4], [3 -1 2 4 -2], [1 2 -1 3 5], ...
        '(x[t] > 0) or (y[t] > 0)');

    % 5. Combined: ((x >= 0) and (y >= 0)) or (z >= 0)
    n = n + eval_3sig(out, 'combined_and_or', ...
        [0 1 2 3 4], ...
        [1 -1 2 -2 3], [2 3 -1 1 -1], [-1 -1 -1 5 -1], ...
        '((x[t] > 0) and (y[t] > 0)) or (z[t] > 0)');

    %% ======== Always (G) ========

    % 6. Sliding window: G[0,2](x >= 0)
    n = n + eval_1sig(out, 'always_sliding', ...
        [0 1 2 3 4], [5 2 8 1 6], ...
        'alw_[0,2] (x[t] > 0)');

    % 7. Boundary clamping: G[0,5](x >= 0) on a short signal
    n = n + eval_1sig(out, 'always_boundary', ...
        [0 1 2], [3 1 4], ...
        'alw_[0,5] (x[t] > 0)');

    %% ======== Eventually (F) ========

    % 8. Sliding window: F[0,2](x >= 0)
    n = n + eval_1sig(out, 'eventually_sliding', ...
        [0 1 2 3 4], [1 5 2 8 3], ...
        'ev_[0,2] (x[t] > 0)');

    % 9. Offset window: F[1,3](x >= 0)
    n = n + eval_1sig(out, 'eventually_offset', ...
        [0 1 2 3 4 5 6], [1 5 2 8 3 1 7], ...
        'ev_[1,3] (x[t] > 0)');

    %% ======== Until ========

    % 10. Basic until: (x >= 0) U[0,3] (y >= 0)
    n = n + eval_2sig(out, 'until_basic', ...
        [0 1 2 3 4], [1 1 1 -1 -1], [-1 -1 2 2 2], ...
        '(x[t] > 0) until_[0,3] (y[t] > 0)');

    % 11. Tight interval: (x >= 0) U[1,2] (y >= 0)
    n = n + eval_2sig(out, 'until_tight', ...
        [0 1 2 3 4], [1 1 1 -1 -1], [-1 -1 2 2 2], ...
        '(x[t] > 0) until_[1,2] (y[t] > 0)');

    % 12. Never satisfied: q always negative
    n = n + eval_2sig(out, 'until_never_sat', ...
        [0 1 2 3 4], [1 1 1 1 1], [-1 -1 -1 -1 -1], ...
        '(x[t] > 0) until_[0,4] (y[t] > 0)');

    %% ======== Nested temporal ========

    t10 = 0:9;
    x10 = [2 -1 3 -2 1 -3 4 0.5 -1 2];

    % 13. G[0,3](F[0,2](x >= 0))
    n = n + eval_1sig(out, 'nested_GF', ...
        t10, x10, ...
        'alw_[0,3] (ev_[0,2] (x[t] > 0))');

    % 14. F[0,2](G[0,1](x >= 0))
    n = n + eval_1sig(out, 'nested_FG', ...
        t10, x10, ...
        'ev_[0,2] (alw_[0,1] (x[t] > 0))');

    %% ======== Non-uniform time sampling ========

    t_nu = [0 0.5 1.5 3 4];

    % 15. G[0,1](x >= 0) on non-uniform times
    n = n + eval_1sig(out, 'nonuniform_always', ...
        t_nu, [0 3 1 4 2], ...
        'alw_[0,1] (x[t] > 0)');

    % 16. Until on non-uniform times
    n = n + eval_2sig(out, 'nonuniform_until', ...
        t_nu, [2 1 -1 1 3], [-1 -1 2 -1 1], ...
        '(x[t] > 0) until_[0,2] (y[t] > 0)');

    %% ======== Interpolation edge cases ========

    % 17. Sparse samples, window between knots
    %     x = [-1, 10, -1], times = [0, 2, 4]
    %     G[0.5,1.5] window falls entirely between sample points
    n = n + eval_1sig(out, 'interp_sparse', ...
        [0 2 4], [-1 10 -1], ...
        'alw_[0.5,1.5] (x[t] > 0)');

    % 18. Window boundaries between adjacent samples
    %     x = [5, -3, 5], times = [0, 1, 2]
    %     G[0.2,0.8] at t=0: window [0.2, 0.8], no sample inside
    n = n + eval_1sig(out, 'interp_boundary', ...
        [0 1 2], [5 -3 5], ...
        'alw_[0.2,0.8] (x[t] > 0)');

    %% ======== Dense signal ========

    % 19. sin(t) with dense sampling
    t_dense = linspace(0, 10, 101);
    x_dense = sin(t_dense);

    n = n + eval_1sig(out, 'dense_sine', ...
        t_dense, x_dense, ...
        'alw_[0,2] (x[t] + 0.5 > 0)');

    %% ======== Until: larger signals / complex transitions ========

    % 20. Until on 10-point signal with multiple sign changes
    n = n + eval_2sig(out, 'until_long_transitions', ...
        0:9, [2 1 3 1 -1 2 1 -2 1 3], [-3 -2 -1 1 2 -1 3 1 -1 2], ...
        '(x[t] > 0) until_[0,4] (y[t] > 0)');

    % 21. Until with non-zero lower bound [1,3] on 8 points
    n = n + eval_2sig(out, 'until_offset_window', ...
        0:7, [3 2 1 2 3 1 -1 2], [-1 -2 -1 4 -1 2 1 -1], ...
        '(x[t] > 0) until_[1,3] (y[t] > 0)');

    %% ======== Nested temporal + boolean ========

    % 22. G[0,2](p and F[0,1](q))
    n = n + eval_2sig(out, 'nested_G_and_F', ...
        0:6, [2 1 3 0.5 2 1 4], [-1 2 -1 3 -2 1 2], ...
        'alw_[0,2] ((x[t] > 0) and (ev_[0,1] (y[t] > 0)))');

    % 23. F[0,2](p or G[0,1](q))
    n = n + eval_2sig(out, 'nested_F_or_G', ...
        0:7, [-1 2 -1 3 -2 1 -1 4], [1 -1 2 -2 3 -1 1 -1], ...
        'ev_[0,2] ((x[t] > 0) or (alw_[0,1] (y[t] > 0)))');

    %% ======== Arithmetic predicates ========

    % 24. x - y >= 1  (includes exact zero robustness at t=2)
    n = n + eval_2sig(out, 'arith_difference', ...
        0:4, [5 3 1 4 2], [1 4 0 2 3], ...
        'x[t] - y[t] - 1 > 0');

    % 25. G[0,1](x + y >= 0) on mostly-negative sum
    n = n + eval_2sig(out, 'arith_sum', ...
        0:4, [-3 1 -2 4 -1], [2 -3 1 -5 2], ...
        'alw_[0,1] (x[t] + y[t] > 0)');

    %% ======== Multi-variable in temporal ========

    % 26. (p and q) U[0,2] r  — three signals
    n = n + eval_3sig(out, 'multivar_until', ...
        0:4, [3 2 -1 1 4], [1 -1 2 3 -2], [-2 -1 1 2 3], ...
        '((x[t] > 0) and (y[t] > 0)) until_[0,2] (z[t] > 0)');

    %% ======== Zero crossings / all-negative ========

    % 27. F[0,2] on signal with multiple zero crossings
    n = n + eval_1sig(out, 'zero_crossings', ...
        0:6, [-2 1 -3 0 2 -1 0.5], ...
        'ev_[0,2] (x[t] > 0)');

    % 28. G[0,2] on entirely negative signal
    n = n + eval_1sig(out, 'all_negative', ...
        0:4, [-5 -3 -1 -4 -2], ...
        'alw_[0,2] (x[t] > 0)');

    % 29. Or with exact zero robustness values
    n = n + eval_2sig(out, 'exact_satisfaction', ...
        0:4, [0 1 0 -1 0], [0 0 1 0 -1], ...
        '(x[t] > 0) or (y[t] > 0)');

    %% ======== Eventually with large offset ========

    % 30. F[2,5] on 10-point signal
    n = n + eval_1sig(out, 'eventually_large_offset', ...
        0:9, [-2 -1 3 -1 5 2 -3 1 4 -1], ...
        'ev_[2,5] (x[t] > 0)');

    %% ======== Deeply nested (3 levels) ========

    % 31. G[0,1](F[0,1](G[0,1](x >= 0)))
    n = n + eval_1sig(out, 'deeply_nested_GFG', ...
        0:7, [1 -0.5 2 -1 0.5 3 -2 1], ...
        'alw_[0,1] (ev_[0,1] (alw_[0,1] (x[t] > 0)))');

    %% ======== Single-point signal ========

    % 32. Degenerate single-point trace
    n = n + eval_1sig(out, 'single_point', ...
        [0], [3], ...
        'x[t] > 0');

    %% ======== out_of_bounds="clamp" ========

    % 33. G[2,4] — window entirely past signal end
    %     Last value is positive; should propagate via clamping.
    n = n + eval_1sig(out, 'clamp_always_past', ...
        [0 1 2], [5 -1 3], ...
        'alw_[2,4] (x[t] > 0)');

    % 34. F[2,4] — window entirely past signal end, last value negative
    %     Clamped -2 is the max in an all-negative extended region.
    n = n + eval_1sig(out, 'clamp_eventually_past', ...
        [0 1 2], [-5 -1 -2], ...
        'ev_[2,4] (x[t] > 0)');

    % 35. G[0,5] with negative tail — clamped value IS the global minimum
    n = n + eval_1sig(out, 'clamp_always_neg_tail', ...
        [0 1 2], [5 3 -2], ...
        'alw_[0,5] (x[t] > 0)');

    % 36. F[0,8] — big window, last value is the max in extended region
    n = n + eval_1sig(out, 'clamp_eventually_big_window', ...
        [0 1 2 3 4], [-2 1 -1 3 -4], ...
        'ev_[0,8] (x[t] > 0)');

    % 37. Until with window starting past signal end
    n = n + eval_2sig(out, 'clamp_until_past', ...
        [0 1 2], [1 2 -1], [-2 3 1], ...
        '(x[t] > 0) until_[3,5] (y[t] > 0)');

    fprintf('\nDone. Generated %d / 37 CSV files.\n', n);

    %% ======== RV 2026 E1 column (extra/experiments/e1_divergence/cache) ========
    % Auto-generated companion (see HANDOFF.md); safe to run repeatedly.
    generate_e1_column();
end


%% ==================== Helper functions ====================

function ok = eval_1sig(out, name, time, x, formula)
%EVAL_1SIG  Evaluate a single-signal test case.
    try
        BrSys = BreachTraceSystem({'x'});
        traj = struct('time', time, 'X', x);
        BrSys.AddTrace(traj);
        save_rob(BrSys, name, formula, time, out);
        ok = 1;
    catch e
        fprintf('  [FAIL] %s: %s\n', name, e.message);
        ok = 0;
    end
end

function ok = eval_2sig(out, name, time, x, y, formula)
%EVAL_2SIG  Evaluate a two-signal test case.
    try
        BrSys = BreachTraceSystem({'x', 'y'});
        traj = struct('time', time, 'X', [x; y]);
        BrSys.AddTrace(traj);
        save_rob(BrSys, name, formula, time, out);
        ok = 1;
    catch e
        fprintf('  [FAIL] %s: %s\n', name, e.message);
        ok = 0;
    end
end

function ok = eval_3sig(out, name, time, x, y, z, formula)
%EVAL_3SIG  Evaluate a three-signal test case.
    try
        BrSys = BreachTraceSystem({'x', 'y', 'z'});
        traj = struct('time', time, 'X', [x; y; z]);
        BrSys.AddTrace(traj);
        save_rob(BrSys, name, formula, time, out);
        ok = 1;
    catch e
        fprintf('  [FAIL] %s: %s\n', name, e.message);
        ok = 0;
    end
end

function save_rob(BrSys, name, formula, time, out)
%SAVE_ROB  Evaluate STL formula and write robustness CSV.
    phi = STL_Formula(name, formula);
    [rob, tau] = STL_Eval(BrSys.Sys, phi, BrSys.P, BrSys.P.traj{1}, time);

    fpath = fullfile(out, [name '.csv']);
    fid = fopen(fpath, 'w');
    fprintf(fid, 'time,robustness\n');
    for i = 1:length(tau)
        fprintf(fid, '%.15g,%.15g\n', tau(i), rob(i));
    end
    fclose(fid);
    fprintf('  [OK] %s -> %d points\n', name, length(tau));
end
