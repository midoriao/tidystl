function generate_ground_truth()
%GENERATE_GROUND_TRUTH  Compute TaLiRo (dp_taliro) robustness for tidystl cases.
%
%   Requires the S-TaLiRo toolbox on the MATLAB path (the toolbox root and
%   its dp_taliro folder). The online Simulink monitor (setup_monitor) is NOT
%   needed and may fail to build without Simulink headers; only the
%   dp_taliro mex is required. See HANDOFF.md for the full setup recipe.
%
%   Usage:
%     >> addpath('/path/to/s-taliro');
%     >> addpath('/path/to/s-taliro/dp_taliro');
%     >> addpath('/path/to/s-taliro/auxiliary');
%     >> cd /path/to/tidystl
%     >> generate_ground_truth
%
%   Writes packages/tidystl-compat/tests/taliro_ground_truth.jsonl (one JSON object per case).
%   dp_taliro returns a SINGLE scalar (the robustness of the whole trace,
%   reported at t=0), so each record holds exactly one point at time 0.
%   This matches tidystl's top-level robustness rho[:, 0].
%
%   Semantic notes (pinned by live dp_taliro runs, R2022b):
%     - Predicates are half-spaces A*x <= b. Robustness is the EUCLIDEAN
%       signed distance (b - A*x)/||A||, i.e. it NORMALIZES by ||A||.
%       For single-variable predicates ||A|| = 1, so the value equals the
%       raw signed distance and matches tidystl/Breach/RTAMT. Multi-variable
%       predicates (e.g. x+y>=0) are divided by ||A|| -- see the
%       `norm_*` cases, which deliberately exercise this divergence.
%     - Timed operators reduce over SAMPLE POINTS inside the time window
%       only; dp_taliro does not interpolate at window endpoints.

    out = fullfile(pwd, 'packages', 'tidystl-compat', 'tests', 'taliro_ground_truth.jsonl');
    records = struct('name', {}, 'time', {}, 'robustness', {});
    fprintf('Generating TaLiRo ground truth -> %s\n\n', out);
    n = 0;

    % Predicate half-space helpers. A predicate "var >= c" over a single
    % signal column j is the half-space -x_j <= -c (A=-1 on column j, b=-c),
    % whose signed distance equals var - c. ge1(j,c) builds it with proj=j.
    ge1 = @(name, j, c) struct('str', name, 'A', -1, 'b', -c, 'proj', j);

    %% ===== Basic predicates (single var; ||A||=1, matches raw value) =====

    % predicate_gte : tidystl  x >= 3   | dp robustness at t0 = x(0)-3
    P = ge1('p', 1, 3);
    [records, ok] = emit(records, 'predicate_gte', 'p', P, [5;2;4;1;6], (0:4)');
    n = n + ok;

    % not_simple : tidystl  not (x >= 0)
    P = ge1('p', 1, 0);
    [records, ok] = emit(records, 'not_simple', '!p', P, [3;-1;2;-4;5], (0:4)');
    n = n + ok;

    % and_two : tidystl  (x >= 0) and (y >= 0)   [columns: x, y]
    P = [ge1('px',1,0), ge1('py',2,0)];
    [records, ok] = emit(records, 'and_two', 'px /\ py', P, [3 1; -1 2; 2 -1; 4 3; -2 5], (0:4)');
    n = n + ok;

    % or_two : tidystl  (x >= 0) or (y >= 0)
    [records, ok] = emit(records, 'or_two', 'px \/ py', P, [3 1; -1 2; 2 -1; 4 3; -2 5], (0:4)');
    n = n + ok;

    %% ===== Always [] =====

    % always_untimed : tidystl  G[0,4] (x >= 0)   | = min over trace = 1
    %   (tidystl has no untimed G; [0,4] spans the whole trace so rho(0) is
    %   identical to the untimed []p value.)
    P = ge1('p', 1, 0);
    [records, ok] = emit(records, 'always_untimed', '[]_[0,4] p', P, [5;2;8;1;6], (0:4)');
    n = n + ok;

    % always_sliding : tidystl  G[0,2] (x >= 0)   | samples-min over [0,2] = 2
    [records, ok] = emit(records, 'always_sliding', '[]_[0,2] p', P, [5;2;8;1;6], (0:4)');
    n = n + ok;

    % always_boundary : tidystl  G[0,5] (x >= 0) on short signal
    [records, ok] = emit(records, 'always_boundary', '[]_[0,5] p', P, [3;1;4], (0:2)');
    n = n + ok;

    %% ===== Eventually <> =====

    % eventually_untimed : tidystl  F[0,4] (x >= 0)   | = max over trace
    [records, ok] = emit(records, 'eventually_untimed', '<>_[0,4] p', P, [1;5;2;8;3], (0:4)');
    n = n + ok;

    % eventually_sliding : tidystl  F[0,2] (x >= 0)
    [records, ok] = emit(records, 'eventually_sliding', '<>_[0,2] p', P, [1;5;2;8;3], (0:4)');
    n = n + ok;

    %% ===== Until =====

    % until_basic : tidystl  (x >= 0) U[0,3] (y >= 0)
    P = [ge1('px',1,0), ge1('py',2,0)];
    [records, ok] = emit(records, 'until_basic', 'px U_[0,3] py', P, ...
        [1 -1; 1 -1; 1 2; -1 2; -1 2], (0:4)');
    n = n + ok;

    % until_untimed : tidystl  (x >= 0) U[0,4] (y >= 0)
    [records, ok] = emit(records, 'until_untimed', 'px U_[0,4] py', P, ...
        [1 -1; 1 -1; 1 2; -1 2; -1 2], (0:4)');
    n = n + ok;

    %% ===== Nested temporal =====

    t10 = (0:9)';
    x10 = [2;-1;3;-2;1;-3;4;0.5;-1;2];
    P = ge1('p', 1, 0);

    % nested_GF : tidystl  G[0,3] (F[0,2] (x >= 0))
    [records, ok] = emit(records, 'nested_GF', '[]_[0,3] (<>_[0,2] p)', P, x10, t10);
    n = n + ok;

    % nested_FG : tidystl  F[0,2] (G[0,1] (x >= 0))
    [records, ok] = emit(records, 'nested_FG', '<>_[0,2] ([]_[0,1] p)', P, x10, t10);
    n = n + ok;

    %% ===== Non-uniform time sampling =====

    t_nu = [0;0.5;1.5;3;4];
    P = ge1('p', 1, 0);
    % nonuniform_always : tidystl  G[0,1] (x >= 0) on non-uniform times
    [records, ok] = emit(records, 'nonuniform_always', '[]_[0,1] p', P, [0;3;1;4;2], t_nu);
    n = n + ok;

    %% ===== Multi-variable predicates: ||A|| normalization (DIVERGENCE) =====
    % These document TaLiRo's Euclidean-distance robustness, which divides
    % by ||A||. A tidystl-compatible backend must replicate the /||A|| factor
    % (or the cases will not match a raw-value semantics).

    % norm_sum : predicate  x + y >= 0  encoded A=[-1 -1], b=0.
    %   At x=3,y=0 the signed distance is (0 - (-3))/sqrt(2) = 3/sqrt(2).
    Pn = struct('str', 'p', 'A', [-1 -1], 'b', 0);
    [records, ok] = emit(records, 'norm_sum', 'p', Pn, [3 0; 3 0], [0;1]);
    n = n + ok;

    % norm_diff : predicate  x - y >= 1  encoded A=[-1 1], b=-1.
    %   At x=5,y=1: (-1 - ([-1 1]*[5;1]))/sqrt(2) = (5-1-1)/sqrt(2)=3/sqrt(2).
    Pd = struct('str', 'p', 'A', [-1 1], 'b', -1);
    [records, ok] = emit(records, 'norm_diff', 'p', Pd, [5 1; 5 1], [0;1]);
    n = n + ok;

    %% ===== Edge cases: empty / past-end windows, until edges =====
    % Pin dp_taliro's samples-only window semantics: no endpoint interpolation,
    % no past-end extension; empty timed window -> +Inf (always) / -Inf (eventually).
    p = ge1('p', 1, 0);

    % Empty window (no samples inside [t+0.2, t+0.8]) -> Inf / -Inf
    [records, ok] = emit(records, 'always_empty_window',     '[]_[0.2,0.8] p', p, [5;-3;5],       [0;1;2]);
    n = n + ok;
    [records, ok] = emit(records, 'eventually_empty_window', '<>_[0.2,0.8] p', p, [5;-3;5],       [0;1;2]);
    n = n + ok;
    % Window partly past trace end: only real in-window samples count (no clamp)
    [records, ok] = emit(records, 'always_past_end',         '[]_[2,4] p',     p, [5;-1;3],       [0;1;2]);
    n = n + ok;
    [records, ok] = emit(records, 'eventually_past_end',     '<>_[2,4] p',     p, [-5;-1;-2],     [0;1;2]);
    n = n + ok;
    [records, ok] = emit(records, 'always_big_window',       '[]_[0,5] p',     p, [5;3;-2],       [0;1;2]);
    n = n + ok;
    [records, ok] = emit(records, 'eventually_big_window',   '<>_[0,8] p',     p, [-2;1;-1;3;-4], (0:4)');
    n = n + ok;
    % Offset windows (lower bound > 0)
    [records, ok] = emit(records, 'always_offset',           '[]_[1,3] p',     p, [1;5;2;8;3],    (0:4)');
    n = n + ok;
    [records, ok] = emit(records, 'eventually_offset',       '<>_[1,3] p',     p, [1;5;2;8;3],    (0:4)');
    n = n + ok;
    % Samples-only vs interpolation probes (interp would give 5; dp_taliro gives the sample)
    [records, ok] = emit(records, 'interp_probe_ev',         '<>_[0,1] p',     p, [0;10],         [0;2]);
    n = n + ok;
    [records, ok] = emit(records, 'interp_probe_alw',        '[]_[0,1] p',     p, [10;0],         [0;2]);
    n = n + ok;

    Pu = [ge1('px', 1, 0), ge1('py', 2, 0)];
    % Until window past trace end -> -Inf; never-satisfied -> negative q value
    [records, ok] = emit(records, 'until_past_end',  'px U_[3,5] py', Pu, [1 -2; 2 3; -1 1],            [0;1;2]);
    n = n + ok;
    [records, ok] = emit(records, 'until_never_sat', 'px U_[0,4] py', Pu, [1 -1;1 -1;1 -1;1 -1;1 -1], (0:4)');
    n = n + ok;
    [records, ok] = emit(records, 'until_tight',     'px U_[1,2] py', Pu, [1 -1; 1 -1; 1 2; -1 2; -1 2], (0:4)');
    n = n + ok;

    [~, order] = sort({records.name});
    records = records(order);
    fid = fopen(out, 'w');
    for k = 1:numel(records)
        fprintf(fid, '%s\n', record_to_json(records(k)));
    end
    fclose(fid);
    fprintf('\nDone. Wrote %d cases -> %s\n', n, out);
end


%% ==================== Helper ====================

function [records, ok] = emit(records, name, phi, Pred, seqS, seqT)
%EMIT  Run dp_taliro on one case; append a single-point record (t=0).
    try
        rob = dp_taliro(phi, Pred, seqS, seqT);
        records(end+1) = struct('name', name, 'time', [0], 'robustness', [rob]); %#ok<AGROW>
        fprintf('  [OK] %-22s rob = %.10g\n', name, rob);
        ok = 1;
    catch e
        fprintf('  [FAIL] %-20s %s\n', name, e.message);
        ok = 0;
    end
end


function s = record_to_json(rec)
%RECORD_TO_JSON  Serialize a record to a JSON line. Emits Infinity/-Infinity/
%   NaN tokens (Python json-compatible) for non-finite robustness and always
%   renders time/robustness as arrays. MATLAB jsonencode maps +/-Inf and NaN
%   all to null (losing the sign), which the Python reader parses as NaN; this
%   keeps round-trips faithful to the recorded fixtures.
    s = sprintf('{"name": %s, "time": %s, "robustness": %s}', ...
        jsonencode(rec.name), num_array_json(rec.time), num_array_json(rec.robustness));
end


function s = num_array_json(v)
    parts = cell(1, numel(v));
    for i = 1:numel(v)
        parts{i} = num_json(v(i));
    end
    s = ['[', strjoin(parts, ', '), ']'];
end


function s = num_json(x)
    if isinf(x)
        if x > 0
            s = 'Infinity';
        else
            s = '-Infinity';
        end
    elseif isnan(x)
        s = 'NaN';
    else
        s = sprintf('%.17g', x);
    end
end
