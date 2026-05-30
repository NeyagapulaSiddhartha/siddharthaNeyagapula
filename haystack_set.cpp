import islpy as isl
from pathlib import Path
import argparse
import re
import io
import time
import argparse
import concurrent.futures
from pathlib import Path
from collections import defaultdict
# ---------------------------------------------------------------------------
# 1.  Correct Lex Relations
# ---------------------------------------------------------------------------

def lex_succ(ctx, n):
    space = isl.Space.set_alloc(ctx, 0, n)
    return isl.Map.lex_lt(space)

def lex_pred(ctx, n):
    space = isl.Space.set_alloc(ctx, 0, n)
    return isl.Map.lex_ge(space)


def lex_succ_equal(ctx: isl.Context, n_dims: int) -> isl.Map:
    space = isl.Space.set_alloc(ctx, 0, n_dims)
    return isl.Map.lex_le(space)


def lex_succ_eq(ctx: isl.Context, n_dims: int) -> isl.Map:
    space = isl.Space.set_alloc(ctx, 0, n_dims)
    return isl.Map.lex_le(space)




import islpy as isl
import islpy as isl



class OpTimer:
    def __init__(self):
        self._total = defaultdict(float)
        self._calls = defaultdict(int)
        self._active_start = {}

    def _start(self, op):
        self._active_start[op] = time.perf_counter()

    def _stop(self, op):
        t0 = self._active_start.pop(op, None)
        if t0 is not None:
            dt = time.perf_counter() - t0
            self._total[op] += dt
            self._calls[op] += 1

    def _timed(self, op, fn, *args):
        self._start(op)
        res = fn(*args)
        self._stop(op)
        return res

    # ---- ISL ops ----
    def coalesce(self, x): return self._timed("coalesce", x.coalesce)
    def subtract(self, a, b): return self._timed("subtract", a.subtract, b)
    def apply_range(self, a, b): return self._timed("apply_range", a.apply_range, b)
    def apply_domain(self, a, b): return self._timed("apply_domain", a.apply_domain, b)
    def range(self, x): return self._timed("range", x.range)
    def reverse(self, x): return self._timed("reverse", x.reverse)
    def union(self, a, b): return self._timed("union", a.union, b)
    def intersect(self, a, b): return self._timed("intersect", a.intersect, b)
    def intersect_domain(self, a, b): return self._timed("intersect", a.intersect_domain, b)
    def lexmin(self, x): return self._timed("lexmin", x.lexmin)
    def count(self, x): return self._timed("count", x.count_val)
    def from_map(self, m): return self._timed("from_map", isl.UnionMap.from_map, m)

    def record_custom(self, name, duration):
        self._total[name] += duration
        self._calls[name] += 1

    def format_summary(self, name):
        buf = io.StringIO()
        buf.write(f"\n{'-'*60}\n")
        buf.write(f"Timing summary: {name}\n")
        buf.write(f"{'-'*60}\n")
        buf.write(f"{'Op':<20} {'Calls':>6} {'Total(s)':>12} {'Avg(ms)':>10}\n")

        for op in sorted(self._total, key=self._total.get, reverse=True):
            total = self._total[op]
            calls = self._calls[op]
            avg = (total / calls) * 1000 if calls else 0
            buf.write(f"{op:<20} {calls:>6} {total:>12.6f} {avg:>10.3f}\n")

        buf.write(f"{'-'*60}\n")
        return buf.getvalue()


def print_miss_points_for_input(forward, input_set_str, ctx, results):
    # input_set is already an isl.UnionSet
    
    # print("Input set string:", input_set_str)
    input_set = isl.UnionSet.read_from_str(ctx, input_set_str)

    # print("Input set:", input_set)
    # rest of your logic stays SAME
    image = input_set.apply(forward)
    # image = forward.apply_domain(input_set)

    # print("Image:", image.coalesce())
    uset = image

    total = 0

    bset_list = uset.get_basic_set_list()

    for i in range(bset_list.n_basic_set()):
        bset = bset_list.get_basic_set(i)

        if bset.is_bounded():
            val = bset.count_val()
            total += val.to_python()
        else:
            print("Unbounded set, skipping")

    # print("Total points in image:", total)
    # print("Cardinality:", total)
    if(total > 2):
        results.append(1)

# ---------------------------------------------------------------------------
# 2.  Algorithm 1 – SingleLevelMisses
# ---------------------------------------------------------------------------

def _get_n_dims(umap: isl.UnionMap) -> int:
    n = [None]
    def _visitor(m):
        if n[0] is None:
            n[0] = m.domain().dim(isl.dim_type.set)
    umap.foreach_map(_visitor)
    return n[0] if n[0] is not None else 0


def lex_prec(ctx, n):
    return isl.Map.lex_lt(isl.Space.set_alloc(ctx, 0, n))

def lex_pred_eq(ctx, n):
    return isl.Map.lex_ge(isl.Space.set_alloc(ctx, 0, n))

def lex_succ(ctx, n_dims):
    return isl.Map.lex_le(isl.Space.set_alloc(ctx, 0, n_dims))

def lex_succ_eq(ctx, n_dims):
    return isl.Map.lex_le(isl.Space.set_alloc(ctx, 0, n_dims))



def single_level_misses(
    ctx: isl.Context,
    prog_refs: isl.UnionMap,
    prog_domain: isl.UnionSet,
    shed: isl.UnionMap,
    linear: isl.UnionMap,
    access_to_line: isl.UnionMap,
    line_id_to_cache_set: isl.UnionMap,
    B: int,
    S: int,
    k: int,
    timer: OpTimer,
):

    t = timer

    prog = t.intersect_domain(shed, prog_domain)

    iter_to_line = t.apply_range(
                       t.apply_range(
                           t.apply_range(
                               t.reverse(prog),
                               prog_refs
                           ),
                           linear
                       ),
                       access_to_line
                   )
    iter_to_line = t.coalesce(iter_to_line)

    iter_to_set = t.coalesce(
        t.apply_range(iter_to_line, line_id_to_cache_set)
    )

    # cold misses: first (in schedule order) iteration to touch each line
    cold_miss_map   = t.lexmin(t.reverse(iter_to_line))
    cold_miss_iters = t.coalesce(t.range(cold_miss_map))

    same_line = t.coalesce(
        t.apply_range(iter_to_line, t.reverse(iter_to_line))
    )

    n_sched_dims = _get_n_dims(same_line)

    lp  = t.from_map(lex_prec(ctx, n_sched_dims))
    lse = t.from_map(lex_succ_eq(ctx, n_sched_dims))
    lpe = t.from_map(lex_pred_eq(ctx, n_sched_dims))

    same_line_suc    = t.coalesce(t.intersect(same_line, lp))
    # same_line_suc_eq = t.coalesce(t.intersect(same_line, lse))

    same_set     = t.coalesce(
        t.apply_range(iter_to_set, t.reverse(iter_to_set))
    )
    same_set_suc = t.coalesce(t.intersect(same_set, lp))


    cold_miss_map   = t.lexmin(t.reverse(iter_to_line))
    cold_miss_iters =   t.range(t.apply_range(cold_miss_map,t.reverse(shed)))

    # ls = isl.UnionMap.from_map(lex_succ(ctx, n_sched_dims))
    # lse = isl.UnionMap.from_map(lex_succ_eq(ctx, n_sched_dims))
    # lpred = isl.UnionMap.from_map(lex_pred(ctx, n_sched_dims))



    # same_line_suc = same_line.intersect(ls).coalesce()
    # same_line_suc_eq = same_line.intersect(lse).coalesce()



    # # STEP 8–9
    # same_set = iter_to_set.apply_range(
    #     iter_to_set.reverse()  
    # ).coalesce()

    # same_set_suc = same_set.intersect(ls).coalesce()


    # print ("  \n\n $$$$$$$$$$$$$$$$$$$$$   all the old steps are done   $$$$$$$$$$$$$$$$$$$$$$$$$$$  \n\n ")
    # print(same_line_suc)
    # print(same_set_suc)

    # print("\n  === Step 8-9: Same set Successors (Strict) ===   \n")

    # print(" \n **************************** same_set_suc.lexmin() **********************************************\n ",same_set_suc.lexmin()," \n **************************** DONE same_set_suc.lexmin() **********************************************\n ",)
    
     # --------------------------------------------------
    # STEP 1: equal_maps
    # --------------------------------------------------
    equal_maps = t.apply_range(
        t.apply_range(
            t.apply_range(
                t.reverse(shed),
                prog_refs
            ),
            t.reverse(prog_refs)
        ),
        shed
    )

    # --------------------------------------------------
    # STEP 2: forward map
    # --------------------------------------------------
    forward = t.apply_range(
        t.reverse(
            t.lexmin(
                t.coalesce(
                    t.intersect(equal_maps, lp)
                )
            )
        ),
        same_set_suc
    )

    # --------------------------------------------------
    # STEP 3: backward map
    # --------------------------------------------------
    backward = t.apply_range(
        t.apply_range(
            shed,
            lpe
        ),
        t.reverse(shed)
    )

    backward = t.apply_range(
        t.apply_range(
            t.reverse(shed),
            backward
        ),
        shed
    )

    # --------------------------------------------------
    # STEP 4: reuse distance
    # --------------------------------------------------
    reuse_dist = t.coalesce(
        t.intersect(forward, backward)
    )





    # equal_maps =(
    #         shed.reverse().apply_range(prog_refs)
    #         .apply_range(prog_refs.reverse())
    #         .apply_range(shed)
    # )
    
    # temp = equal_maps.intersect(ls).coalesce().lexmin().reverse().apply_range(same_set_suc)
    # # temp2 = equal_maps.lexmin().reverse().apply_range(lse).intersect_range(shed.range())
    # forward = temp

    # # print(" foward ",equal_maps.lexmin())

    #     # print(" \n ************************Forward map:******************************************",ii, m," \n************************Forward map:****************************************** \n")
    # # same_set_suc.foreach_map(pr)
    # # print(" <------------------------- the forward map is --------------------------------> ")
    # # print(forward)
    # # print("\n=== Step 8-9: Same set Successors (Forward) ===")
    # # print(forward)

    


    # backward = (shed.apply_range(lpred)
    #                         .apply_range(shed.reverse()))
    # # print(" #################################################### 0000000000000000000000000000000 ########################################################\n")
    # # print(backward)
    # # print(" #################################################### 111111111111111111111111111111111 ########################################################\n")
    # # print(shed.reverse().apply_range(backward))
    # # print(" #################################################### 22222222222222222222222222222222222222 ########################################################\n")
    # # print(backward.apply_range(shed))
    
    # backward = shed.reverse().apply_range(backward).apply_range(shed)

    # reuse_dist = forward.intersect(backward).coalesce()


    # print_miss_points_for_input(reuse_dist, " { [3, 2, 2, 2 ,0] } ", ctx, [])

    print("\n=== Reuse Distance (Iteration Space) ===")

    # print(reuse_dist)
    
    count = 0

    results = []
    shed.range().foreach_point(
        lambda p: print_miss_points_for_input(
            reuse_dist,
             p.to_str(),
            ctx,
            results
        )
    )
    
    set_list = cold_miss_iters.get_set_list()

    total = sum(
        set_list.get_at(i).count_val().to_python()
        for i in range(set_list.n_set())
    )
    print("Results: ", len(results) + total)

    return reuse_dist , count

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def extract_linear_map(text: str) -> str | None:
    """
    Extract the { ... } block under:
    === LINEARIZATION MAP ===
    Works even if formatting/newlines change.
    """
    start = text.find("=== LINEARIZATION MAP ===")
    if start == -1:
        return None

    sub = text[start:]

    # Find first '{'
    brace_start = sub.find('{')
    if brace_start == -1:
        return None

    # Extract balanced braces
    depth = 0
    for i, ch in enumerate(sub[brace_start:], start=brace_start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return sub[brace_start:i+1]

    return None

# ===========================================================================
# Parsing haystack output files
# ===========================================================================

def extract_isl_string(text: str, tag: str) -> str | None:
    pattern = rf'\[START\]\s*\[{re.escape(tag)}\](.*?)\[Done\]\s*\[{re.escape(tag)}\]'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    raw = match.group(1).strip()
    brace_start = raw.find('{')
    if brace_start == -1:
        return None

    depth = 0
    for i, ch in enumerate(raw[brace_start:], start=brace_start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return raw[brace_start: i + 1]
    return None


def parse_haystack_file(filepath: Path):
    """Returns (parsed_dict, missing_tags_list)."""
    text = filepath.read_text()

    accesses      = extract_isl_string(text, "Accesses")
    access_domain = extract_isl_string(text, "AccessDomain")
    schedule      = extract_isl_string(text, "Schedule")
    linear        = extract_linear_map(text)


    missing = [tag for tag, v in [("Accesses", accesses),
                                   ("AccessDomain", access_domain),
                                   ("Schedule", schedule),
                                   ("Linear", linear)] if v is None]
    if missing:
        return None, missing

    return {
        "accesses":      accesses,
        "access_domain": access_domain,
        "schedule":      schedule,
        "linear":        linear,
    }, []


# ===========================================================================
# Build linear and cache-set maps
# ===========================================================================

def build_linear_map(ctx: isl.Context, prog_refs: isl.UnionMap) -> isl.UnionMap:
    arrays = {}

    def _collect(m: isl.Map):
        rng  = m.range()
        name = rng.get_tuple_name()
        ndim = rng.dim(isl.dim_type.set)
        if name not in arrays:
            arrays[name] = ndim

    prog_refs.foreach_map(_collect)

    STRIDE = 100_000
    parts  = []
    base   = 0

    for name, ndim in sorted(arrays.items()):
        if ndim == 1:
            idx_vars    = "i0"
            linear_expr = "i0"
        elif ndim == 2:
            idx_vars    = "i0, i1"
            linear_expr = f"i0*{STRIDE} + i1"
        elif ndim == 3:
            idx_vars    = "i0, i1, i2"
            linear_expr = f"i0*{STRIDE*STRIDE} + i1*{STRIDE} + i2"
        else:
            idx_vars    = ", ".join(f"i{k}" for k in range(ndim))
            linear_expr = " + ".join(
                f"i{k}*{STRIDE**(ndim-1-k)}" for k in range(ndim)
            )

        parts.append(
            f"{name}[{idx_vars}] -> [linear] : linear = {base} + {linear_expr}"
        )
        base += STRIDE ** ndim

    isl_str = "{ " + "; ".join(parts) + " }"
    return isl.UnionMap.read_from_str(ctx, isl_str)


def make_cache_map_str(num_sets: int) -> str:
    return (f"{{ [lid] -> [cset] : exists q : "
            f"lid = {num_sets}*q + cset and 0 <= cset < {num_sets}; }}")


# ===========================================================================
# Per-benchmark runner  — designed to run in a worker process
# ===========================================================================

def _write_output(output_dir: Path, benchmark_name: str, content: str) -> Path:
    """Write captured output to <output_dir>/<benchmark_name>_result.txt"""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{benchmark_name}_result.txt"
    out_file.write_text(content)
    return out_file


def run_benchmark(
    filepath: Path,
    output_dir: Path,
    num_sets: int,
    line_size: int,
    assoc: int,
    verbose: bool = False,
) -> tuple[str, Path]:
    """
    Run the full cache miss analysis for one benchmark.

    All console output is captured into a string, written to
      <output_dir>/<benchmark_stem>_result.txt
    and also returned to the main process for console printing.

    Returns: (output_string, output_file_path)
    """
    buf  = io.StringIO()
    name = filepath.stem

    def log(msg=""):
        buf.write(msg + "\n")

    log(f"\n{'='*60}")
    log(f"Benchmark : {name}")
    log(f"File      : {filepath}")

    # --- parse ---------------------------------------------------------------
    parsed, missing = parse_haystack_file(filepath)
    if missing:
        log(f"  [SKIP] Missing sections {missing} in {filepath.name}")
        out_path = _write_output(output_dir, name, buf.getvalue())
        return buf.getvalue(), out_path

    if verbose:
        log("\n[Accesses ISL]\n"     + parsed["accesses"])
        log("\n[AccessDomain ISL]\n" + parsed["access_domain"])
        log("\n[Schedule ISL]\n"     + parsed["schedule"])

    # Each worker creates its own isl.Context — ISL contexts are not
    # process-safe to share across fork boundaries.
    ctx = isl.Context()
    B   = line_size
    S   = num_sets
    K   = assoc

    try:
        prog_refs   = isl.UnionMap.read_from_str(ctx, parsed["accesses"])
        prog_domain = isl.UnionSet.read_from_str(ctx, parsed["access_domain"])
        shed        = isl.UnionMap.read_from_str(ctx, parsed["schedule"])
        linear      = isl.UnionMap.read_from_str(ctx, parsed["linear"])
    except Exception as e:
        log(f"  [ERROR] ISL parse failed: {e}")
        out_path = _write_output(output_dir, name, buf.getvalue())
        return buf.getvalue(), out_path

    # linear = build_linear_map(ctx, prog_refs)
    linear = isl.UnionMap.read_from_str(ctx, parsed["linear"])

    if verbose:
        log("\n[Linear map]\n" + str(linear))

    access_to_line = isl.UnionMap.read_from_str(
        ctx,
        f"{{[linear] -> [lid] : exists r : linear = {B}*lid + r and 0 <= r < {B};}}"
    )
    line_id_to_cache_set = isl.UnionMap.read_from_str(
        ctx, make_cache_map_str(num_sets)
    )

    timer = OpTimer()

    try:
        misses, miss_count = single_level_misses(
            ctx, prog_refs, prog_domain, shed,
            linear, access_to_line, line_id_to_cache_set,
            B, S, K,
            timer=timer,
        )
    except Exception as e:
        log(f"  [ERROR] Cache analysis failed: {e}")
        out_path = _write_output(output_dir, name, buf.getvalue())
        return buf.getvalue(), out_path

    log(f"\nResult -> Miss count: {miss_count}  "
        f"(sets={num_sets}, line_size={B}, assoc={K})")
    log(timer.format_summary(name))

    out_path = _write_output(output_dir, name, buf.getvalue())
    return buf.getvalue(), out_path


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cache miss analysis from haystack outputs (parallel)"
    )
    parser.add_argument("--input_dir",  required=True,
                        help="Directory containing haystack output .txt files")
    parser.add_argument("--output_dir", default="./cache_results",
                        help="Directory for per-benchmark result files "
                             "(default: ./cache_results)")
    parser.add_argument("--num_sets",   type=int, required=True,
                        help="Number of cache sets")
    parser.add_argument("--line_size",  type=int, default=1,
                        help="Cache line size in elements (default: 1)")
    parser.add_argument("--assoc",      type=int, default=2,
                        help="Cache associativity K (default: 2)")
    parser.add_argument("--workers",    type=int, default=None,
                        help="Parallel worker processes "
                             "(default: cpu_count)")
    parser.add_argument("--benchmark",  default=None,
                        help="Run only this benchmark (substring match)")
    parser.add_argument("--verbose",    action="store_true",
                        help="Include parsed ISL strings in output files")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"ERROR: {input_dir} is not a directory.")
        return

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"ERROR: No .txt files found in {input_dir}")
        return

    if args.benchmark:
        txt_files = [f for f in txt_files if args.benchmark in f.stem]
        if not txt_files:
            print(f"No files matching benchmark '{args.benchmark}'")
            return

    print(f"Found   : {len(txt_files)} benchmark file(s)")
    print(f"Config  : sets={args.num_sets}, line_size={args.line_size}, assoc={args.assoc}")
    print(f"Workers : {args.workers or 'auto (cpu_count)'}")
    print(f"Output  : {output_dir.resolve()}")
    print()

    wall_start = time.perf_counter()

    # ProcessPoolExecutor — true parallelism; each worker gets its own
    # Python interpreter + ISL context, so there are no GIL or ISL
    # thread-safety concerns.
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:

        future_to_file = {
            executor.submit(
                run_benchmark,
                f,
                output_dir,
                args.num_sets,
                args.line_size,
                args.assoc,
                args.verbose,
            ): f
            for f in txt_files
        }

        for future in concurrent.futures.as_completed(future_to_file):
            filepath = future_to_file[future]
            try:
                output, out_path = future.result()
                print(output, end="")
                print(f"  [SAVED] → {out_path}")
            except Exception as exc:
                print(f"\n[FATAL] {filepath.stem}: {exc}")

    wall_elapsed = time.perf_counter() - wall_start
    print(f"\n{'='*60}")
    print(f"All done.  Total wall time: {wall_elapsed:.3f}s")


if __name__ == "__main__":
    main()
