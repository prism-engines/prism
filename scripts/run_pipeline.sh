#!/bin/bash
# PRISM Pipeline Runner
# =====================
# Runs the complete PRISM pipeline for a domain
#
# Usage: ./scripts/run_pipeline.sh <domain>
# Example: ./scripts/run_pipeline.sh cheme

set -e  # Exit on error

DOMAIN=${1:-cheme}

echo "=============================================="
echo "PRISM Pipeline for: $DOMAIN"
echo "=============================================="
echo

# Check if domain is configured
echo "Checking configuration..."
python -m prism.assessments.config $DOMAIN
echo

read -p "Continue with pipeline? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Step 1: Signal Vector
echo
echo "=============================================="
echo "Step 1/4: Computing signal vectors..."
echo "=============================================="
python -m prism.entry_points.signal_vector --domain $DOMAIN

# Step 2: Geometry
echo
echo "=============================================="
echo "Step 2/4: Computing geometry..."
echo "=============================================="
python -m prism.entry_points.geometry --domain $DOMAIN

# Step 3: Laplace (includes modes)
echo
echo "=============================================="
echo "Step 3/4: Computing Laplace field..."
echo "=============================================="
python -m prism.entry_points.laplace --domain $DOMAIN

# Step 4: Assessment
echo
echo "=============================================="
echo "Step 4/4: Running assessment..."
echo "=============================================="
python -m prism.assessments.run --domain $DOMAIN

echo
echo "=============================================="
echo "Pipeline complete for: $DOMAIN"
echo "=============================================="
