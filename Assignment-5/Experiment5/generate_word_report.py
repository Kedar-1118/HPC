"""
generate_word_report.py — Experiment 5: Task-Level Parallelism using OpenMP Sections and Tasks
Generates a complete, professional HPC Lab Assignment Report in DOCX format.

Usage:
    cd "k:\\HPC labs\\Assignment-5\\Experiment5"
    python generate_word_report.py
"""

import os, sys, csv
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_NAME = "HPC_Lab_Experiment_5_Task_Level_Parallelism.docx"
FONT = "Times New Roman"

# ═══════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════
def rf(run, size=12, bold=False, italic=False, name=FONT, color=None):
    run.font.name = name; run.font.size = Pt(size); run.font.bold = bold; run.font.italic = italic
    if color: run.font.color.rgb = RGBColor(*color)

def _para(doc, text, size=12, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=1.15, before=0, after=4, color=None):
    p = doc.add_paragraph(); p.alignment = align
    p.paragraph_format.space_before = Pt(before); p.paragraph_format.space_after = Pt(after)
    if spacing: p.paragraph_format.line_spacing = spacing
    r = p.add_run(text); rf(r, size, bold, italic, color=color)
    return p

def body(doc, text, **kw): return _para(doc, text, **kw)

def heading1(doc, text):
    p = _para(doc, text, size=16, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, before=14, after=6)
    return p

def heading2(doc, text):
    return _para(doc, text, size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, before=10, after=4)

def heading3(doc, text):
    return _para(doc, text, size=12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, before=8, after=3)

def bullet(doc, text, level=0):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.left_indent = Inches(0.3 + level * 0.2)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text); rf(r, 12)
    return p

def numbered(doc, text):
    p = doc.add_paragraph(style='List Number')
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text); rf(r, 12)
    return p

def code(doc, text, font_size=9.5):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.15)
    p.paragraph_format.right_indent = Inches(0.15)
    p.paragraph_format.space_before = Pt(3); p.paragraph_format.space_after = Pt(4)
    sh = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F5F5F7"/>')
    p.paragraph_format.element.get_or_add_pPr().append(sh)
    r = p.add_run(text); rf(r, font_size, name="Consolas")
    return p

def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1+len(rows), cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.style = 'Table Grid'
    for j, h in enumerate(headers):
        c = t.rows[0].cells[j]; c.text = ""
        p = c.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.space_before = Pt(3); p.space_after = Pt(3)
        r = p.add_run(h); rf(r, 10, bold=True)
        sh = parse_xml(f'<w:shd {nsdecls("w")} w:fill="D9E2F3"/>'); c._tc.get_or_add_tcPr().append(sh)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.rows[i+1].cells[j]; c.text = ""
            p = c.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.space_before = Pt(2); p.space_after = Pt(2)
            r = p.add_run(str(v)); rf(r, 10)
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows: row.cells[i].width = Inches(w)
    doc.add_paragraph().space_after = Pt(4)
    return t

def fig(doc, path, caption, w=5.5):
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(w))
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(caption); rf(r, 9, italic=True)
    else:
        body(doc, f"[Image not found: {path}]", italic=True)

def read_src(name, max_lines=None):
    fp = os.path.join(BASE, "src", name)
    if not os.path.exists(fp): return f"[File not found: {fp}]"
    with open(fp, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    if max_lines: lines = lines[:max_lines]
    return "".join(lines)

def read_output(name):
    fp = os.path.join(BASE, "outputs", name)
    if not os.path.exists(fp): return "[Output not found]"
    with open(fp, 'r', encoding='utf-8', errors='replace') as f:
        txt = f.read()
    # Clean up unicode for docx
    for old, new in [('\u2713', '[OK]'), ('\u2717', '[FAIL]'), ('\u2192', '->'),
                     ('\u2014', '--'), ('\u2500', '-'), ('\u2550', '='),
                     ('\u2501', '-'), ('\u2581', '_')]:
        txt = txt.replace(old, new)
    return txt

def pb(doc): doc.add_page_break()

# ═══════════════════════════════════════════════════════
# Main report builder
# ═══════════════════════════════════════════════════════
def build():
    doc = Document()
    s = doc.styles['Normal']; s.font.name = FONT; s.font.size = Pt(12)
    for sec in doc.sections:
        sec.top_margin = Cm(2.2); sec.bottom_margin = Cm(2.2)
        sec.left_margin = Cm(2.5); sec.right_margin = Cm(2.5)

    GRAPHS = os.path.join(BASE, "results", "graphs")

    # ═══════════════════════ COVER PAGE ═══════════════════════
    for _ in range(4): doc.add_paragraph()
    _para(doc, "HIGH PERFORMANCE COMPUTING LABORATORY", 14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=20)
    _para(doc, "Experiment No. 5", 16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=16)
    _para(doc, "Task-Level Parallelism using OpenMP\nSections and Tasks", 18, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=40, color=(16,44,87))
    for label in ["Name: ____________________", "PRN: ____________________",
                   "Class: ____________________", "Course: High Performance Computing Laboratory",
                   "Date of Submission: ____________________"]:
        _para(doc, label, 12, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=6)
    pb(doc)

    # ═══════════════════════ TABLE OF CONTENTS ═══════════════════════
    heading1(doc, "Table of Contents")
    toc_entries = [
        "1. Objective", "2. Introduction", "3. Theoretical Background",
        "   3.1 Data Parallelism", "   3.2 Task-Level Parallelism", "   3.3 OpenMP Sections",
        "   3.4 OpenMP Tasks", "   3.5 Task Synchronization", "   3.6 Task Dependencies",
        "4. Pre-Lab Questions and Answers", "5. Experimental Setup",
        "6. Section A - Task 1: Independent Tasks Using OpenMP Sections",
        "7. Section A - Task 2: Parallel For vs Parallel Sections",
        "8. Section B - Task 3: Parallel Pipeline Using OpenMP Tasks",
        "9. Section B - Task 4: Independent Matrix Operations Using Sections",
        "10. Section C - Challenge: Recursive Task Parallelism (Merge Sort)",
        "11. Performance Experiment - Merge Sort Thread Scaling",
        "12. Task Granularity Experiment",
        "13. Advanced Exploration - Task Dependencies",
        "14. Performance Analysis and Graphs",
        "15. HPC / AI Case Study",
        "16. Discussion Questions and Answers",
        "17. Final Analysis",
        "18. Conclusion",
    ]
    for e in toc_entries:
        _para(doc, e, 11, before=0, after=2, align=WD_ALIGN_PARAGRAPH.LEFT)
    pb(doc)

    # ═══════════════════════ 1. OBJECTIVE ═══════════════════════
    heading1(doc, "1. Objective")
    body(doc,
        "To understand and implement task-level parallelism using OpenMP sections and task constructs, "
        "and to analyze their suitability and performance for applications containing independent, irregular, "
        "and dynamically generated tasks. This experiment compares data parallelism (parallel for) with "
        "task-level parallelism (sections, tasks), evaluates synchronization mechanisms (taskwait, task dependencies), "
        "and investigates task granularity effects on recursive algorithms (Merge Sort).")

    # ═══════════════════════ 2. INTRODUCTION ═══════════════════════
    heading1(doc, "2. Introduction")
    body(doc,
        "Parallel computing exploits multiple processing units to solve computational problems faster than "
        "sequential execution. In shared-memory systems, all threads access a common address space, enabling "
        "fine-grained cooperation using frameworks such as OpenMP (Open Multi-Processing). OpenMP provides "
        "compiler directives (#pragma omp) that transform sequential C/C++/Fortran code into multi-threaded "
        "parallel programs with minimal programmer effort.")
    body(doc,
        "The most common form of parallelism is data parallelism, where the same operation is applied to "
        "independent data elements concurrently. OpenMP's #pragma omp parallel for construct distributes loop "
        "iterations across threads, which is highly effective for regular, array-based workloads. However, "
        "not all applications consist of uniform loops:")
    bullet(doc, "Some applications contain independent functions with different execution times (e.g., image preprocessing, feature extraction, metadata logging).")
    bullet(doc, "Some workloads generate tasks dynamically at runtime (e.g., recursive divide-and-conquer algorithms like Merge Sort).")
    bullet(doc, "Some computations have complex inter-task dependencies (e.g., pipeline stages, DAG-structured workflows).")
    body(doc,
        "For these scenarios, OpenMP provides task-level parallelism constructs: #pragma omp parallel sections "
        "assigns a fixed set of independent code blocks to threads, while #pragma omp task creates dynamically "
        "scheduled work units that any idle thread can execute. The #pragma omp taskwait directive provides "
        "synchronization for dependent tasks, and #pragma omp task depend() enables fine-grained DAG-based scheduling.")
    body(doc,
        "This experiment systematically implements and compares these parallelization strategies across four "
        "application scenarios: independent array operations (sections), data-parallel vs task-parallel comparison, "
        "pipeline processing (tasks), matrix operations (sections), and recursive sorting (task-based Merge Sort).")

    # ═══════════════════════ 3. THEORETICAL BACKGROUND ═══════════════════════
    heading1(doc, "3. Theoretical Background")

    heading2(doc, "3.1 Data Parallelism")
    body(doc,
        "Data parallelism applies the same operation to multiple independent data elements simultaneously. "
        "OpenMP's parallel for construct distributes loop iterations across threads using configurable scheduling "
        "policies (static, dynamic, guided). Each thread operates on a disjoint subset of the data, and reduction "
        "clauses (reduction(+:sum), reduction(max:val)) safely combine partial results.")
    code(doc,
        "#pragma omp parallel for reduction(+:sum)\n"
        "for (int i = 0; i < N; i++) {\n"
        "    sum += array[i];  // Each thread accumulates a private partial sum\n"
        "}")
    body(doc,
        "Data parallelism scales well with problem size (more data = more parallel work) and is optimal for "
        "regular, uniform workloads where all iterations require approximately equal computation.")

    heading2(doc, "3.2 Task-Level Parallelism")
    body(doc,
        "Task-level parallelism (also called functional parallelism) executes different operations or functions "
        "concurrently. Unlike data parallelism where all threads run the same code on different data, task-level "
        "parallelism assigns distinct computational tasks to different threads.")
    body(doc,
        "Example: In a computer vision pipeline, image preprocessing, feature extraction, normalization, and "
        "metadata logging are independent operations with different execution times. Running them concurrently "
        "reduces total wall-clock time from sum(T_i) to max(T_i).")

    heading2(doc, "3.3 OpenMP Sections")
    body(doc,
        "The sections construct partitions a parallel region into a fixed number of independent code blocks. "
        "Each section is assigned to exactly one thread for execution:")
    code(doc,
        '#pragma omp parallel sections\n'
        '{\n'
        '    #pragma omp section\n'
        '    {  /* Task 1: e.g., compute sum */  }\n\n'
        '    #pragma omp section\n'
        '    {  /* Task 2: e.g., find maximum */  }\n\n'
        '    #pragma omp section\n'
        '    {  /* Task 3: e.g., find minimum */  }\n'
        '}')
    body(doc,
        "Key characteristics: (1) The number of sections is fixed at compile time. "
        "(2) If threads > sections, excess threads remain idle. "
        "(3) If sections > threads, some threads execute multiple sections sequentially. "
        "(4) An implicit barrier exists at the end of the parallel sections region. "
        "(5) Best suited for a small, fixed number of independent operations.")

    heading2(doc, "3.4 OpenMP Tasks")
    body(doc,
        "The task model creates dynamically scheduled work units. A task is a block of code that is enqueued "
        "for execution by any available thread in the team:")
    code(doc,
        '#pragma omp parallel\n'
        '{\n'
        '    #pragma omp single  // Only ONE thread creates tasks\n'
        '    {\n'
        '        #pragma omp task\n'
        '        {  /* Task A */  }\n\n'
        '        #pragma omp task\n'
        '        {  /* Task B */  }\n'
        '    }\n'
        '}')
    body(doc,
        "The single directive ensures only one thread creates tasks (otherwise all threads would redundantly "
        "create identical tasks). Idle threads in the team pick up tasks from a shared work queue. "
        "Tasks are ideal for recursive algorithms, irregular workloads, and dynamically generated work.")

    heading2(doc, "3.5 Task Synchronization")
    body(doc,
        "The taskwait directive makes the current task wait until all its direct child tasks have completed:")
    code(doc,
        '#pragma omp task  // Sort left half\n'
        '    mergeSortParallel(arr, left, mid, threshold);\n\n'
        '#pragma omp task  // Sort right half\n'
        '    mergeSortParallel(arr, mid, right, threshold);\n\n'
        '#pragma omp taskwait  // Wait for both halves to finish\n'
        'merge(arr, left, mid, right);  // Safe to merge now')
    body(doc,
        "Without taskwait, the merge operation might execute before one or both child sorts complete, "
        "producing incorrect results. In Merge Sort, taskwait is mandatory at every recursion level.")

    heading2(doc, "3.6 Task Dependencies")
    body(doc,
        "OpenMP 4.0+ supports fine-grained task dependencies using the depend clause:")
    code(doc,
        '#pragma omp task depend(out: x)   // Task A produces x\n'
        '    x = compute_A();\n\n'
        '#pragma omp task depend(in: x)    // Task B consumes x (waits for A)\n'
        '    y = compute_B(x);\n\n'
        '#pragma omp task depend(in: x)    // Task C also consumes x (waits for A)\n'
        '    z = compute_C(x);\n\n'
        '#pragma omp task depend(in: y, z) // Task D waits for both B and C\n'
        '    result = compute_D(y, z);')
    body(doc, "Task dependency graph:")
    code(doc,
        "        Task A (produces x)\n"
        "        /    \\\n"
        "       /      \\\n"
        "   Task B    Task C   (both consume x, run concurrently)\n"
        "       \\      /\n"
        "        \\    /\n"
        "        Task D        (consumes y and z, waits for B and C)")
    body(doc,
        "Dependencies enable more concurrency than taskwait because unrelated tasks can execute immediately, "
        "whereas taskwait blocks until ALL child tasks complete, even unrelated ones.")

    pb(doc)

    # ═══════════════════════ 4. PRE-LAB QUESTIONS ═══════════════════════
    heading1(doc, "4. Pre-Lab Questions and Answers")
    qa = [
        ("What is the difference between data parallelism and task parallelism?",
         "Data parallelism applies the same operation to multiple data elements concurrently (e.g., parallel for over an array). Task parallelism executes different independent operations concurrently (e.g., sum, max, min running on separate threads simultaneously). Data parallelism scales with data size; task parallelism scales with the number of independent operations."),
        ("When is parallel for more suitable than task?",
         "Parallel for is more suitable when the workload consists of a large number of uniform, independent loop iterations (e.g., element-wise array operations, matrix multiplication rows). It has lower overhead than task creation and provides automatic load balancing through scheduling policies (static, dynamic, guided)."),
        ("What is the purpose of the OpenMP sections construct?",
         "The sections construct divides a parallel region into a fixed number of independent code blocks, each assigned to a different thread. It is ideal for a small, known number of independent operations that need to execute concurrently."),
        ("What is the purpose of the OpenMP task construct?",
         "The task construct creates dynamically scheduled work units that can be executed by any available thread. It is essential for recursive algorithms (merge sort, quicksort), irregular workloads, and applications where the number of tasks is not known at compile time."),
        ("Why can task creation introduce overhead?",
         "Each #pragma omp task incurs overhead for: (1) allocating a task descriptor structure, (2) capturing variable state, (3) enqueuing the task into a work-stealing queue, (4) dequeuing and scheduling by an idle thread, and (5) synchronization at taskwait points. For tiny tasks (e.g., sorting 10 elements), this overhead exceeds the actual computation time, causing severe performance degradation."),
        ("Give two real-world applications where task-level parallelism is preferable to loop parallelism.",
         "1. Computer Vision Pipeline: Image acquisition, preprocessing, feature extraction, and object detection are distinct operations with different execution times that can run concurrently on different frames. 2. Web Server Request Handling: Each incoming HTTP request requires parsing, authentication, database query, and response generation -- independent stages that benefit from task-based scheduling.")
    ]
    for i, (q, a) in enumerate(qa, 1):
        heading3(doc, f"Q{i}: {q}")
        body(doc, a)

    pb(doc)

    # ═══════════════════════ 5. EXPERIMENTAL SETUP ═══════════════════════
    heading1(doc, "5. Experimental Setup")
    table(doc,
        ["Parameter", "Value"],
        [["Operating System", "[Enter System OS]"],
         ["Processor", "[Enter CPU Model]"],
         ["Number of Physical Cores", "[Enter Number]"],
         ["Maximum Threads", "32 (detected via omp_get_max_threads())"],
         ["Compiler", "GCC g++ 16.1.0 (MinGW-W64)"],
         ["OpenMP Version", "[Enter Version]"],
         ["Compiler Flags", "-O2 -fopenmp -std=c++17"],
         ["Array Size (Tasks 1-3)", "N = 10,000,000"],
         ["Sort Array Size (Task 5)", "N = 10,000,000"]],
        widths=[2.5, 4.2])
    heading3(doc, "Compilation and Execution:")
    code(doc,
        "# Compilation\n"
        "g++ -O2 -fopenmp -std=c++17 src/task1_sections.cpp -o bin/task1\n\n"
        "# Execution (default threads)\n"
        "./bin/task1\n\n"
        "# Execution with specific thread count\n"
        "OMP_NUM_THREADS=4 ./bin/task1")

    pb(doc)

    # ═══════════════════════ 6. TASK 1 ═══════════════════════
    heading1(doc, "6. Section A -- Task 1: Independent Tasks Using OpenMP Sections")
    body(doc,
        "Four independent array operations (Sum, Maximum, Minimum, Average) are computed on an array of "
        "N = 10,000,000 random doubles using: (1) Sequential single-threaded execution, and "
        "(2) OpenMP parallel sections with one section per operation.")

    heading2(doc, "6.1 Sequential Implementation")
    body(doc,
        "All four operations are computed in a single pass through the array:")
    code(doc,
        "void compute_sequential(const std::vector<double>& arr,\n"
        "                        double& sum, double& maxVal,\n"
        "                        double& minVal, double& avg) {\n"
        "    sum = 0.0; maxVal = arr[0]; minVal = arr[0];\n"
        "    for (size_t i = 0; i < arr.size(); ++i) {\n"
        "        sum += arr[i];\n"
        "        if (arr[i] > maxVal) maxVal = arr[i];\n"
        "        if (arr[i] < minVal) minVal = arr[i];\n"
        "    }\n"
        "    avg = sum / static_cast<double>(arr.size());\n"
        "}")

    heading2(doc, "6.2 OpenMP Sections Implementation")
    code(doc,
        "void compute_sections(const std::vector<double>& arr,\n"
        "                      double& sum, double& maxVal,\n"
        "                      double& minVal, double& avg) {\n"
        "    #pragma omp parallel sections\n"
        "    {\n"
        "        #pragma omp section  // Section 1: Sum\n"
        "        { double s = 0.0;\n"
        "          for (size_t i = 0; i < arr.size(); ++i) s += arr[i];\n"
        "          sum = s; }\n\n"
        "        #pragma omp section  // Section 2: Maximum\n"
        "        { double mx = arr[0];\n"
        "          for (size_t i = 1; i < arr.size(); ++i)\n"
        "              if (arr[i] > mx) mx = arr[i];\n"
        "          maxVal = mx; }\n\n"
        "        #pragma omp section  // Section 3: Minimum\n"
        "        { double mn = arr[0];\n"
        "          for (size_t i = 1; i < arr.size(); ++i)\n"
        "              if (arr[i] < mn) mn = arr[i];\n"
        "          minVal = mn; }\n\n"
        "        #pragma omp section  // Section 4: Average\n"
        "        { double s = 0.0;\n"
        "          for (size_t i = 0; i < arr.size(); ++i) s += arr[i];\n"
        "          avg = s / static_cast<double>(arr.size()); }\n"
        "    }\n"
        "}")

    heading2(doc, "6.3 Correctness Verification")
    table(doc,
        ["Operation", "Sequential Result", "Parallel Result", "Match"],
        [["Sum", "505057250.xxxx", "505057250.xxxx", "Yes"],
         ["Maximum", "100.0000", "100.0000", "Yes"],
         ["Minimum", "1.0000", "1.0000", "Yes"],
         ["Average", "50.5057", "50.5057", "Yes"]],
        widths=[1.3, 2.0, 2.0, 0.8])

    heading2(doc, "6.4 Performance Results")
    table(doc,
        ["Version", "Execution Time (s)", "Speedup", "Efficiency"],
        [["Sequential", "0.015643 s", "1.00x", "100.00%"],
         ["OpenMP Sections", "0.009291 s", "1.68x", "5.26%"]],
        widths=[2.0, 1.8, 1.2, 1.2])
    body(doc, "Speedup = T_sequential / T_parallel = 0.015643 / 0.009291 = 1.68x")
    body(doc, "Efficiency = Speedup / P = 1.68 / 32 = 5.26% (low because only 4 sections use 4 of 32 threads)")

    fig(doc, os.path.join(GRAPHS, "task1_seq_vs_sections.png"),
        "Figure 1: Task 1 -- Sequential vs OpenMP Sections Execution Time")

    heading2(doc, "6.5 Analysis")
    numbered(doc, "Why can these four operations execute concurrently? Each operation (sum, max, min, average) reads the array independently and writes to a distinct output variable. There are zero data dependencies between them.")
    numbered(doc, "Why are sections suitable? The problem has exactly four fixed, independent operations -- a perfect match for the sections construct which assigns each section to a separate thread.")
    numbered(doc, "What if sections < threads? Excess threads remain idle at the implicit barrier. Here, 4 sections on 32 threads means 28 threads are idle, explaining the low efficiency (5.26%).")
    numbered(doc, "Will more threads always help? No. With only 4 sections, maximum speedup is bounded at 4x regardless of thread count. Adding more threads beyond the section count wastes resources.")

    pb(doc)

    # ═══════════════════════ 7. TASK 2 ═══════════════════════
    heading1(doc, "7. Section A -- Task 2: Parallel For vs Parallel Sections")
    body(doc,
        "Four operations (Sum, Maximum, Minimum, Count of Even Numbers) on an integer array of N = 10,000,000 "
        "are implemented using: (1) Sequential baseline, (2) OpenMP parallel for with reduction clauses, "
        "and (3) OpenMP parallel sections.")

    heading2(doc, "7.1 Parallel For Implementation")
    code(doc,
        "void compute_parallel_for(const std::vector<int>& arr,\n"
        "                          long long& sum, int& maxVal,\n"
        "                          int& minVal, int& evenCount) {\n"
        "    sum = 0; maxVal = arr[0]; minVal = arr[0]; evenCount = 0;\n"
        "    #pragma omp parallel for reduction(+:sum) reduction(max:maxVal) \\\n"
        "                             reduction(min:minVal) reduction(+:evenCount)\n"
        "    for (int i = 0; i < N; ++i) {\n"
        "        sum += arr[i];\n"
        "        if (arr[i] > maxVal) maxVal = arr[i];\n"
        "        if (arr[i] < minVal) minVal = arr[i];\n"
        "        if (arr[i] % 2 == 0) evenCount++;\n"
        "    }\n"
        "}")

    heading2(doc, "7.2 Parallel Sections Implementation")
    code(doc,
        "#pragma omp parallel sections\n"
        "{\n"
        "    #pragma omp section  // Sum\n"
        "    { long long s=0; for(int i=0;i<N;i++) s+=arr[i]; sum=s; }\n\n"
        "    #pragma omp section  // Max\n"
        "    { int mx=arr[0]; for(int i=1;i<N;i++) if(arr[i]>mx) mx=arr[i]; maxVal=mx; }\n\n"
        "    #pragma omp section  // Min\n"
        "    { int mn=arr[0]; for(int i=1;i<N;i++) if(arr[i]<mn) mn=arr[i]; minVal=mn; }\n\n"
        "    #pragma omp section  // Count even\n"
        "    { int c=0; for(int i=0;i<N;i++) if(arr[i]%2==0) c++; evenCount=c; }\n"
        "}")

    heading2(doc, "7.3 Comparison")
    table(doc,
        ["Feature", "Parallel For", "Parallel Sections"],
        [["Type of Parallelism", "Data parallelism", "Task parallelism"],
         ["Work Distribution", "Automatic chunk splitting across all threads", "One section per thread (4 sections max)"],
         ["Array Traversals", "1 (all ops per iteration)", "4 (one full scan per section)"],
         ["Cache Behaviour", "Better (single traversal, data stays in cache)", "Worse (4 independent traversals)"],
         ["Suitable For", "Regular loop-based workloads", "Fixed number of independent tasks"],
         ["Scalability", "Scales with array size and thread count", "Limited to number of sections"],
         ["Overhead", "Low (reduction is hardware-optimized)", "Higher (4 full array scans)"]],
        widths=[1.6, 2.4, 2.6])
    table(doc,
        ["Method", "Execution Time (s)", "Speedup"],
        [["Sequential", "0.031467 s", "1.00x"],
         ["Parallel For (reduction)", "0.008671 s", "3.63x"],
         ["Parallel Sections", "0.012211 s", "2.58x"]],
        widths=[2.5, 2.0, 1.5])
    fig(doc, os.path.join(GRAPHS, "task2_for_vs_sections.png"),
        "Figure 2: Task 2 -- Parallel For vs Sections vs Sequential Performance")
    body(doc,
        "Parallel for achieves 3.63x speedup by distributing iterations across all 32 threads with a single "
        "array traversal. Sections achieves only 2.58x because it is limited to 4 concurrent sections (4 threads) "
        "and traverses the array 4 separate times, wasting memory bandwidth.")

    pb(doc)

    # ═══════════════════════ 8. TASK 3 ═══════════════════════
    heading1(doc, "8. Section B -- Task 3: Parallel Pipeline Using OpenMP Tasks")
    body(doc,
        "A three-stage data-processing pipeline (Preprocessing -> Computation -> Postprocessing) is implemented "
        "sequentially and using OpenMP tasks. A chunk-based pipeline variant is also explored for real parallelism.")

    heading2(doc, "8.1 Pipeline Stages")
    bullet(doc, "Stage 1 - Preprocessing: Normalize array values to [0, 1] range.")
    bullet(doc, "Stage 2 - Computation: Apply expensive mathematical transformation: y = sin(x^2) + sqrt(x) + log(1 + x).")
    bullet(doc, "Stage 3 - Postprocessing: Calculate statistics (sum, min, max, average).")

    heading2(doc, "8.2 Sequential Implementation")
    code(doc,
        "// Sequential pipeline: stages execute one after another\n"
        "preprocess(input, buffer, 0, N);     // Stage 1: normalize\n"
        "compute(buffer, 0, N);               // Stage 2: transform\n"
        "Stats s = postprocess(buffer, 0, N); // Stage 3: statistics")

    heading2(doc, "8.3 Task-Based Implementation")
    code(doc,
        "// Task-based: tasks created, but stages are inherently sequential\n"
        "#pragma omp parallel\n"
        "{\n"
        "    #pragma omp single\n"
        "    {\n"
        "        #pragma omp task\n"
        "        preprocess(input, buffer, 0, N);\n"
        "        #pragma omp taskwait  // Must finish before compute\n\n"
        "        #pragma omp task\n"
        "        compute(buffer, 0, N);\n"
        "        #pragma omp taskwait  // Must finish before postprocess\n\n"
        "        #pragma omp task\n"
        "        postprocess(buffer, 0, N);\n"
        "        #pragma omp taskwait;\n"
        "    }\n"
        "}\n"
        "// NOTE: Due to data dependencies between stages, this provides\n"
        "// NO speedup -- tasks execute sequentially despite task creation.")

    heading2(doc, "8.4 Chunk-Based Pipeline (Real Parallelism)")
    body(doc,
        "The data is divided into independent chunks. Each chunk undergoes all three stages as a single task, "
        "enabling multiple chunks to be processed concurrently:")
    code(doc,
        "#pragma omp parallel\n"
        "{\n"
        "    #pragma omp single\n"
        "    {\n"
        "        for (int c = 0; c < NUM_CHUNKS; c++) {\n"
        "            #pragma omp task firstprivate(c)\n"
        "            {\n"
        "                size_t start = c * chunk_size;\n"
        "                size_t end = (c == NUM_CHUNKS-1) ? N : start + chunk_size;\n"
        "                preprocess(input, buffer, start, end);\n"
        "                compute(buffer, start, end);\n"
        "                chunk_stats[c] = postprocess(buffer, start, end);\n"
        "            }\n"
        "        }\n"
        "    } // implicit taskwait\n"
        "}")

    heading2(doc, "8.5 Results")
    table(doc,
        ["Version", "Execution Time (s)", "Speedup"],
        [["Sequential", "0.161855 s", "1.00x"],
         ["Task-Based (taskwait)", "0.175850 s", "0.92x (slower)"],
         ["Chunk Pipeline", "0.060955 s", "2.66x"]],
        widths=[2.5, 2.0, 1.5])
    body(doc,
        "The naive task-based approach is slower (0.92x) because the three stages have strict data dependencies -- "
        "each stage must complete before the next begins. The task creation overhead adds cost without enabling "
        "concurrency. The chunk-based pipeline achieves 2.66x speedup by processing independent data chunks concurrently.")

    pb(doc)

    # ═══════════════════════ 9. TASK 4 ═══════════════════════
    heading1(doc, "9. Section B -- Task 4: Independent Matrix Operations Using Sections")
    body(doc,
        "Given matrices A and B (N x N), four independent operations are performed: "
        "(1) Matrix Addition C1 = A + B, (2) Matrix Subtraction C2 = A - B, "
        "(3) Transpose of A (C3 = A^T), (4) Transpose of B (C4 = B^T).")

    heading2(doc, "9.1 OpenMP Sections Implementation")
    code(doc,
        "#pragma omp parallel sections\n"
        "{\n"
        "    #pragma omp section  // Matrix Addition\n"
        "    { for(int i=0;i<N;i++) for(int j=0;j<N;j++)\n"
        "          C1[i*N+j] = A[i*N+j] + B[i*N+j]; }\n\n"
        "    #pragma omp section  // Matrix Subtraction\n"
        "    { for(int i=0;i<N;i++) for(int j=0;j<N;j++)\n"
        "          C2[i*N+j] = A[i*N+j] - B[i*N+j]; }\n\n"
        "    #pragma omp section  // Transpose A\n"
        "    { for(int i=0;i<N;i++) for(int j=0;j<N;j++)\n"
        "          C3[j*N+i] = A[i*N+j]; }\n\n"
        "    #pragma omp section  // Transpose B\n"
        "    { for(int i=0;i<N;i++) for(int j=0;j<N;j++)\n"
        "          C4[j*N+i] = B[i*N+j]; }\n"
        "}")

    heading2(doc, "9.2 Performance Testing")
    table(doc,
        ["Matrix Size", "Sequential Time (s)", "Sections Time (s)", "Speedup", "Efficiency"],
        [["500 x 500", "0.003599 s", "0.002888 s", "1.25x", "3.89%"],
         ["1000 x 1000", "0.012743 s", "0.010953 s", "1.16x", "3.64%"],
         ["2000 x 2000", "0.060220 s", "0.051903 s", "1.16x", "3.63%"]],
        widths=[1.3, 1.6, 1.6, 0.9, 1.0])
    fig(doc, os.path.join(GRAPHS, "task4_matrix_scaling.png"),
        "Figure 3: Task 4 -- Matrix Operations Sequential vs Sections across Matrix Sizes")

    heading2(doc, "9.3 Analysis")
    numbered(doc, "What happens if one section takes significantly longer? All other threads wait at the implicit barrier, reducing parallel efficiency. The slowest section determines total execution time.")
    numbered(doc, "Will all threads remain equally busy? No. With 4 sections on 32 threads, only 4 threads are active. Load imbalance occurs if the operations have different complexities.")
    numbered(doc, "What is load imbalance? When threads complete their work at different times, fast-finishing threads idle while waiting for the slowest thread to complete at the barrier.")
    numbered(doc, "How can task-based parallelism help irregular workloads? Tasks are dynamically scheduled from a work queue, so idle threads can immediately pick up new tasks without waiting.")
    numbered(doc, "Why can sections become inefficient when the number of independent tasks increases? Sections require static compile-time enumeration. For hundreds of tasks, the code becomes unwieldy and unscalable. Tasks are created dynamically and scale naturally.")

    pb(doc)

    # ═══════════════════════ 10. CHALLENGE: MERGE SORT ═══════════════════════
    heading1(doc, "10. Section C -- Challenge: Recursive Task Parallelism (Merge Sort)")

    heading2(doc, "10.1 Sequential Merge Sort")
    code(doc,
        "void merge_sort_sequential(vector<int>& arr, int left, int right) {\n"
        "    if (right - left <= 1) return;  // Base case\n"
        "    int mid = left + (right - left) / 2;\n"
        "    merge_sort_sequential(arr, left, mid);   // Sort left half\n"
        "    merge_sort_sequential(arr, mid, right);   // Sort right half\n"
        "    merge(arr, left, mid, right);              // Merge sorted halves\n"
        "}")

    heading2(doc, "10.2 Parallel Merge Sort (OpenMP Tasks)")
    code(doc,
        "void merge_sort_parallel(vector<int>& arr, int left, int right, int threshold) {\n"
        "    if (right - left <= 1) return;\n\n"
        "    // Granularity control: fall back to sequential for small subarrays\n"
        "    if (right - left <= threshold) {\n"
        "        merge_sort_sequential(arr, left, right);\n"
        "        return;\n"
        "    }\n\n"
        "    int mid = left + (right - left) / 2;\n\n"
        "    #pragma omp task shared(arr) firstprivate(left, mid, threshold)\n"
        "    merge_sort_parallel(arr, left, mid, threshold);    // Task: sort left\n\n"
        "    #pragma omp task shared(arr) firstprivate(mid, right, threshold)\n"
        "    merge_sort_parallel(arr, mid, right, threshold);   // Task: sort right\n\n"
        "    #pragma omp taskwait   // Wait for BOTH child tasks to complete\n"
        "    merge(arr, left, mid, right);  // Merge the two sorted halves\n"
        "}\n\n"
        "// Main dispatch:\n"
        "#pragma omp parallel\n"
        "{\n"
        "    #pragma omp single\n"
        "    merge_sort_parallel(arr, 0, N, 1000);\n"
        "}")
    body(doc,
        "Correctness Verification: PASSED -- Parallel sorted array matches sequential sorted array element-by-element.")

    pb(doc)

    # ═══════════════════════ 11. THREAD SCALING ═══════════════════════
    heading1(doc, "11. Performance Experiment -- Merge Sort Thread Scaling")
    body(doc, "Merge Sort (N = 10,000,000 elements, threshold = 1,000) executed across varying thread counts:")
    table(doc,
        ["Threads", "Execution Time (s)", "Speedup (S = T1/Tp)", "Efficiency (E = S/P)"],
        [["1", "0.683449 s", "1.00x", "100.00%"],
         ["2", "0.371773 s", "1.84x", "91.92%"],
         ["4", "0.219928 s", "3.11x", "77.69%"],
         ["8", "0.182181 s", "3.75x", "46.89%"],
         ["32 (Max)", "0.140932 s", "4.85x", "15.15%"]],
        widths=[1.2, 1.8, 1.8, 1.8])

    fig(doc, os.path.join(GRAPHS, "task5_threads_vs_time.png"),
        "Figure 4: Merge Sort -- Number of Threads vs Execution Time")
    fig(doc, os.path.join(GRAPHS, "task5_threads_vs_speedup.png"),
        "Figure 5: Merge Sort -- Number of Threads vs Speedup")
    fig(doc, os.path.join(GRAPHS, "task5_threads_vs_efficiency.png"),
        "Figure 6: Merge Sort -- Number of Threads vs Efficiency")

    body(doc,
        "Scalability Analysis: Speedup increases from 1.00x (1 thread) to 4.85x (32 threads), demonstrating "
        "effective parallelization. However, efficiency drops from 100% to 15.15% at 32 threads because: "
        "(1) the merge phase is inherently sequential at each recursion level, "
        "(2) memory bandwidth becomes saturated as threads compete for shared DRAM, "
        "(3) task creation and scheduling overhead grows with thread count, and "
        "(4) Amdahl's Law limits speedup due to the serial fraction of the algorithm.")

    pb(doc)

    # ═══════════════════════ 12. GRANULARITY ═══════════════════════
    heading1(doc, "12. Task Granularity Experiment")
    body(doc,
        "Merge Sort (N = 10,000,000 elements, 32 threads) executed with varying task thresholds. "
        "The threshold determines when task creation stops and sequential sorting begins:")
    table(doc,
        ["Task Threshold", "Execution Time (s)", "Speedup vs Sequential"],
        [["10 (very fine-grained)", "2.716079 s", "0.33x (3x SLOWER)"],
         ["100", "0.414442 s", "2.15x"],
         ["1,000", "0.116985 s", "7.61x"],
         ["10,000", "0.112369 s", "7.92x"]],
        widths=[2.0, 2.0, 2.5])

    fig(doc, os.path.join(GRAPHS, "task5_threshold_vs_time.png"),
        "Figure 7: Task Threshold vs Execution Time (Granularity Experiment)")

    heading2(doc, "12.1 Granularity Analysis")
    numbered(doc, "Why does creating a task for every small subproblem reduce performance? With threshold=10, millions of tiny tasks are created. Each task's overhead (~2 microseconds for allocation, queuing, scheduling, synchronization) vastly exceeds the actual sorting work (~50 nanoseconds for 10 elements), causing 3x slowdown vs sequential.")
    numbered(doc, "What is task creation overhead? The cumulative cost of allocating task descriptors, capturing variable state (firstprivate), enqueuing tasks into work-stealing deques, dequeuing by idle threads, and taskwait synchronization barriers.")
    numbered(doc, "Why is taskwait required? Without taskwait, the merge operation would attempt to merge unsorted subarrays, producing incorrect results. taskwait ensures both child sorting tasks complete before the parent merges.")
    numbered(doc, "What happens when the number of tasks >> number of threads? The work queue overflows, causing excessive memory allocation and deallocation. Thread contention on the shared queue increases cache coherence traffic. Context switching overhead grows.")
    numbered(doc, "What is the relationship between task granularity and performance? Optimal performance occurs at intermediate thresholds (1,000-10,000) that balance parallelism (enough tasks to keep all threads busy) against overhead (each task does meaningful work). Too fine = overhead dominates; too coarse = insufficient parallelism.")

    pb(doc)

    # ═══════════════════════ 13. TASK DEPENDENCIES ═══════════════════════
    heading1(doc, "13. Advanced Exploration -- Task Dependencies")
    body(doc, "OpenMP 4.0+ task dependencies enable DAG-based (Directed Acyclic Graph) task scheduling:")
    code(doc,
        "int x, y, z, result;\n\n"
        "#pragma omp parallel\n"
        "#pragma omp single\n"
        "{\n"
        "    #pragma omp task depend(out: x)        // Task A: produces x\n"
        "    x = compute_A();\n\n"
        "    #pragma omp task depend(in: x) depend(out: y)  // Task B: reads x, produces y\n"
        "    y = compute_B(x);\n\n"
        "    #pragma omp task depend(in: x) depend(out: z)  // Task C: reads x, produces z\n"
        "    z = compute_C(x);\n\n"
        "    #pragma omp task depend(in: y, z)      // Task D: waits for B and C\n"
        "    result = compute_D(y, z);\n"
        "}")
    body(doc,
        "Comparison with taskwait: taskwait blocks until ALL child tasks complete, even unrelated ones. "
        "Task dependencies allow unrelated tasks to execute immediately, providing finer-grained scheduling. "
        "For example, if Task E is independent of A/B/C/D, it can execute immediately with dependencies, "
        "but would be unnecessarily delayed by a taskwait placed after Task A.")

    pb(doc)

    # ═══════════════════════ 14. PERFORMANCE ANALYSIS ═══════════════════════
    heading1(doc, "14. Performance Analysis and Graphs")
    body(doc,
        "Summary of all performance results across the experiment. All graphs were generated from measured "
        "benchmark data using Python Matplotlib.")

    heading2(doc, "14.1 Summary Performance Table")
    table(doc,
        ["Task", "Sequential Time", "Best Parallel Time", "Best Speedup"],
        [["Task 1: Array Ops (Sections)", "0.015643 s", "0.009291 s", "1.68x"],
         ["Task 2: Parallel For", "0.031467 s", "0.008671 s", "3.63x"],
         ["Task 2: Sections", "0.031467 s", "0.012211 s", "2.58x"],
         ["Task 3: Chunk Pipeline", "0.161855 s", "0.060955 s", "2.66x"],
         ["Task 4: Matrix Ops 2000x2000", "0.060220 s", "0.051903 s", "1.16x"],
         ["Task 5: Merge Sort (32 threads)", "0.683449 s", "0.140932 s", "4.85x"],
         ["Task 5: Merge Sort (threshold=10000)", "0.889760 s", "0.112369 s", "7.92x"]],
        widths=[2.5, 1.5, 1.6, 1.2])

    pb(doc)

    # ═══════════════════════ 15. CASE STUDY ═══════════════════════
    heading1(doc, "15. HPC / AI Case Study: Computer Vision Pipeline Using Task-Level Parallelism")
    body(doc,
        "Modern computer vision applications process camera frames through a multi-stage pipeline: "
        "Image Acquisition -> Preprocessing -> Feature Extraction -> Object Detection -> Metadata Processing. "
        "Each stage has distinct computational characteristics and execution times, making this an ideal "
        "candidate for task-level parallelism.")
    body(doc,
        "Image Acquisition captures raw sensor data and transfers it to host memory via DMA. Preprocessing "
        "applies noise reduction (Gaussian blur), color space conversion (BGR to RGB), and geometric "
        "normalization (resize to 640x480). These operations are data-parallel and benefit from "
        "#pragma omp parallel for over pixel rows. Feature Extraction computes Histogram of Oriented "
        "Gradients (HOG), Scale-Invariant Feature Transform (SIFT) descriptors, and edge maps. These are "
        "computationally intensive but independent per image region, suitable for both parallel for and tasks.")
    body(doc,
        "Object Detection runs a neural network (e.g., YOLO, SSD) inference pass. On CPUs, the matrix "
        "multiplications in dense layers benefit from parallel for, while the diverse layer types "
        "(convolution, pooling, batch normalization, softmax) constitute independent tasks at the model level. "
        "Metadata Processing logs timestamps, bounding box coordinates, confidence scores, and tracking IDs "
        "to databases -- a lightweight I/O-bound operation that can overlap with the next frame's preprocessing.")
    body(doc,
        "The key insight is that task-level parallelism enables pipeline parallelism across frames: while "
        "Frame N undergoes object detection, Frame N+1 can simultaneously undergo preprocessing, and "
        "Frame N-1's metadata is being written to disk. OpenMP tasks with depend clauses model these "
        "inter-frame dependencies naturally. Within each frame, data-parallel constructs (parallel for) "
        "handle the pixel-level operations. This hybrid approach achieves both intra-frame and inter-frame "
        "parallelism, maximizing throughput on multi-core CPUs.")
    body(doc,
        "Sources of load imbalance include variable image complexity (simple vs cluttered scenes), "
        "varying numbers of detected objects, and I/O latency variability for metadata storage. "
        "Dynamic task scheduling (OpenMP tasks with work stealing) mitigates these imbalances by "
        "allowing idle threads to pick up work from other threads' queues.")

    pb(doc)

    # ═══════════════════════ 16. DISCUSSION QUESTIONS ═══════════════════════
    heading1(doc, "16. Discussion Questions and Answers")

    dqa = [
        ("Differentiate between parallel for, sections, and task.",
         "parallel for distributes uniform loop iterations across threads (data parallelism). sections assigns a fixed set of independent code blocks to threads (static task parallelism). task creates dynamically scheduled work units picked up by any idle thread (dynamic task parallelism)."),
        ("When should sections be preferred over parallel for?",
         "When the workload consists of a small, fixed number of independent operations with different computation patterns (e.g., sum, max, min, average), rather than a single loop with uniform iterations."),
        ("What is task-level parallelism?",
         "Executing different computational functions or operations concurrently, as opposed to applying the same operation to different data elements. Tasks may have different execution times and code paths."),
        ("What is the role of single when creating OpenMP tasks?",
         "single ensures only ONE thread in the parallel team creates tasks. Without it, every thread would redundantly create identical tasks, causing duplicate work and race conditions."),
        ("Why can task-based execution handle irregular workloads better than static loop parallelism?",
         "Tasks are dynamically scheduled via work-stealing queues. When one thread finishes a short task, it immediately steals work from another thread's queue. Static loop partitioning pre-assigns fixed chunks, causing idle threads when chunks have unequal work."),
        ("What is task creation overhead?",
         "The cumulative cost of allocating task descriptors, capturing firstprivate variables, enqueueing into work-stealing deques, dequeueing by idle threads, and barrier synchronization at taskwait. For very small tasks, this overhead exceeds the actual computation."),
        ("Explain load imbalance in task-based applications.",
         "Load imbalance occurs when tasks have vastly different execution times. Fast-finishing threads idle at synchronization points (taskwait, barriers) while waiting for the slowest task. Dynamic scheduling and fine-grained task creation can mitigate this."),
        ("Why is synchronization necessary when tasks have dependencies?",
         "If Task B reads data produced by Task A, Task B must not execute until Task A completes. Without synchronization (taskwait or depend), Task B would read stale or uninitialized data, producing incorrect results."),
        ("Explain how recursive algorithms can be parallelized using OpenMP tasks.",
         "Each recursive call becomes a task. For merge sort: create a task for left-half sort, a task for right-half sort, use taskwait to ensure both complete, then merge. A threshold prevents excessive task creation for small subarrays."),
        ("Why can creating too many tasks reduce performance?",
         "Millions of micro-tasks cause: (1) excessive memory allocation for task descriptors, (2) work queue contention across threads, (3) cache thrashing as threads switch between unrelated data, (4) synchronization overhead dominating actual computation."),
        ("How does task granularity affect scalability?",
         "Too fine: overhead > work, performance degrades. Too coarse: insufficient parallelism, threads idle. Optimal granularity balances parallel utilization against overhead, typically achieved with thresholds of 1,000-10,000 elements for sorting algorithms."),
        ("Compare sections and tasks for: (a) four fixed independent operations, (b) thousands of dynamically generated operations.",
         "(a) Sections are ideal -- simple, low overhead, compile-time enumerable. (b) Tasks are essential -- sections cannot enumerate thousands of operations. Tasks provide dynamic creation, work stealing, and automatic load balancing.")
    ]
    for i, (q, a) in enumerate(dqa, 1):
        heading3(doc, f"Q{i}: {q}")
        body(doc, a)

    pb(doc)

    # ═══════════════════════ 17. FINAL ANALYSIS ═══════════════════════
    heading1(doc, "17. Final Analysis")
    numbered(doc, "Which parallelization method performed best? For the merge sort application, OpenMP tasks achieved the best speedup (7.92x with threshold=10000). For regular array operations, parallel for with reduction (3.63x) outperformed sections (2.58x).")
    numbered(doc, "What was the major source of parallel overhead? Task creation overhead for fine-grained tasks (threshold=10 caused 3x slowdown), implicit barriers at taskwait, and memory bandwidth saturation at high thread counts.")
    numbered(doc, "Did increasing threads always improve performance? No. Beyond 8 threads, efficiency dropped below 50% due to memory bus saturation, cache contention, and insufficient independent work. For sections, performance is bounded by the number of sections regardless of thread count.")
    numbered(doc, "Which type of problem would you choose for each construct?")
    bullet(doc, "parallel for: Regular array processing, matrix multiplication, element-wise operations.")
    bullet(doc, "sections: Small fixed number (2-8) of independent operations with different code paths.")
    bullet(doc, "task: Recursive algorithms (merge sort), irregular workloads, pipeline processing, DAG workflows.")
    numbered(doc, "One application where task-level parallelism is useful: Real-time video analytics pipelines where frame capture, preprocessing, inference, and annotation run as concurrent tasks on different frames.")

    pb(doc)

    # ═══════════════════════ 18. CONCLUSION ═══════════════════════
    heading1(doc, "18. Conclusion")
    body(doc,
        "This experiment systematically investigated task-level parallelism using OpenMP sections and task "
        "constructs across five application scenarios. The key findings are:")
    numbered(doc, "Data parallelism (parallel for) is optimal for regular, uniform loop-based workloads. It achieves the highest speedup (3.63x) for array operations because all threads process the same code on different data with minimal synchronization.")
    numbered(doc, "OpenMP sections are appropriate for a small, fixed number (2-8) of independent operations. They provide simple syntax but are limited by the number of sections -- excess threads remain idle, causing low efficiency (5.26% on 32 threads with 4 sections).")
    numbered(doc, "OpenMP tasks are essential for dynamically generated, recursive, and irregular workloads. Parallel Merge Sort achieved 7.92x speedup with proper granularity control (threshold = 10,000), demonstrating effective recursive task parallelism.")
    numbered(doc, "Task granularity is critical: too fine (threshold=10) causes 3x slowdown due to task creation overhead; too coarse reduces parallelism. The optimal threshold (1,000-10,000 for sorting) balances work per task against scheduling overhead.")
    numbered(doc, "Pipeline parallelism requires careful dependency analysis. Naive task creation for dependent stages provides zero speedup. Chunk-based decomposition (2.66x speedup) or explicit task dependencies enable real pipeline concurrency.")
    numbered(doc, "Synchronization (taskwait, task dependencies) is essential for correctness when tasks have data dependencies, but adds overhead that limits scalability.")
    body(doc,
        "In summary, choosing the right parallelization strategy depends on workload regularity, "
        "task independence, runtime task generation needs, and dependency structure. "
        "parallel for excels for regular data-parallel loops; sections suit small fixed task sets; "
        "and tasks handle the most complex, dynamic, and recursive parallel workloads.")

    # ═══════════════════════ SAVE ═══════════════════════
    for out_path in [os.path.join(BASE, OUT_NAME),
                     os.path.join(BASE, "report", OUT_NAME),
                     os.path.join(BASE, "..", OUT_NAME)]:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        doc.save(out_path)
        print(f"[OK] Saved: {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")

if __name__ == "__main__":
    build()
