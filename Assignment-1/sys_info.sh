#!/bin/bash
# Hardware & Software System Probe Script for HPC Assignment 1
# Target File: system_configuration.txt

TARGET_LOG="system_configuration.txt"

echo "Querying system hardware and software configuration..."

{
    echo "================================================================="
    echo "         HPC LAB ASSIGNMENT 1 - SYSTEM CONFIGURATION REPORT      "
    echo "================================================================="
    echo "Generated on: $(date)"
    echo "-----------------------------------------------------------------"
    echo ""

    echo "[1] PROCESSOR MODEL"
    echo "-------------------"
    if command -v lscpu &> /dev/null; then
        CPU_NAME=$(lscpu | grep "Model name:" | sed 's/Model name:[ \t]*//')
        echo "Model Name: ${CPU_NAME}"
        else
        CPU_NAME=$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | xargs)
        echo "Model Name: ${CPU_NAME}"
    fi
    echo ""

    echo "[2] CPU CORES & LOGICAL PROCESSORS"
    echo "----------------------------------"
    if command -v lscpu &> /dev/null; then
        NUM_SOCKETS=$(lscpu | grep "Socket(s):" | awk '{print $2}')
        CORES_PER_SOCKET=$(lscpu | grep "Core(s) per socket:" | awk '{print $4}')
        THREADS_PER_CORE=$(lscpu | grep "Thread(s) per core:" | awk '{print $4}')
        TOTAL_LOGICAL_CPUS=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
        
        NUM_PHYSICAL_CORES=$((NUM_SOCKETS * CORES_PER_SOCKET))
        echo "Physical Cores:     ${NUM_PHYSICAL_CORES}"
        echo "Logical Processors: ${TOTAL_LOGICAL_CPUS}"
        echo "Threads per Core:   ${THREADS_PER_CORE}"
    else
        TOTAL_LOGICAL_CPUS=$(nproc)
        NUM_PHYSICAL_CORES=$(grep -c "^processor" /proc/cpuinfo)
        echo "Physical Cores:     ${NUM_PHYSICAL_CORES} (estimated)"
        echo "Logical Processors: ${TOTAL_LOGICAL_CPUS}"
    fi
    echo ""

    echo "[3] SYSTEM RAM"
    echo "--------------"
    if command -v free &> /dev/null; then
        free -h | grep "Mem:" | awk '{print "Total Memory (RAM): " $2 " (Available: " $7 ")"}'
    else
        grep MemTotal /proc/meminfo
    fi
    echo ""

    echo "[4] GCC COMPILER VERSION"
    echo "------------------------"
    if command -v gcc &> /dev/null; then
        gcc --version | head -n 1
    else
        echo "GCC Compiler: Not installed or not found in PATH"
    fi
    echo ""

    echo "[5] OPENMP VERIFICATION"
    echo "-----------------------"
    if command -v gcc &> /dev/null; then
        TEST_SRC=$(mktemp /tmp/omp_check_XXXXXX.c)
        cat << 'EOF' > "$TEST_SRC"
#include <stdio.h>
#include <omp.h>
int main(void) {
    #pragma omp parallel
    {
        #pragma omp single
        printf("OpenMP is supported! Version spec date: %d (Active Threads: %d)\n", _OPENMP, omp_get_num_threads());
    }
    return 0;
}
EOF
        if gcc -fopenmp "$TEST_SRC" -o /tmp/omp_check_bin &> /dev/null; then
            echo "OpenMP Support: VERIFIED & WORKING"
            /tmp/omp_check_bin
            rm -f /tmp/omp_check_bin
        else
            echo "OpenMP Support: FAILED (-fopenmp flag error)"
        fi
        rm -f "$TEST_SRC"
    else
        echo "OpenMP Support: Cannot test (GCC not installed)"
    fi
    echo ""

    echo "[6] LINUX KERNEL VERSION & OS"
    echo "-----------------------------"
    echo "Kernel Version: $(uname -r)"
    echo "Full System:   $(uname -a)"
    if [ -f /etc/os-release ]; then
        echo "OS Distribution: $(grep PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d '"')"
    fi
    echo "Environment:   Windows Subsystem for Linux (WSL)"
    echo ""
    echo "================================================================="
} | tee "$TARGET_LOG"

echo "System configuration report written to ${TARGET_LOG}."

