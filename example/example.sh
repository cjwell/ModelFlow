#!/bin/bash

# ModelFlow Example Script
# This script demonstrates three common use cases of ModelFlow
# Make sure you have the required input files: ARG.tsv, CLSI.tsv, MIC.tsv

echo "=========================================="
echo "ModelFlow Example Workflow"
echo "=========================================="
echo ""

## Example 1: Basic model training for AMK susceptibility prediction
## -i: Input feature table (ARG gene presence/absence or abundance)
## -g: Phenotype file with CLSI interpretation results
## -o: Output directory for AMK results
## Purpose: Train a machine learning model to predict amikacin (AMK) susceptibility 
##          based on antimicrobial resistance genes (ARGs)
echo "Example 1: Training AMK susceptibility prediction model..."
python ../ModelFlow.py -i ARG.tsv -g CLSI -o AMK_CLSI
echo "Completed: AMK model saved to AMK_CLSI/"
echo ""

## Example 2: Model training with specified target column for FEP prediction
## -gc 2: Use the second column from CLSI file (FEP results) as prediction target
## Purpose: Predict cefepime (FEP) susceptibility using the same ARG features
##          Demonstrates how to select different phenotypes from the same file
echo "Example 2: Training FEP susceptibility prediction model (using column 2)..."
python ../ModelFlow.py -i ARG.tsv -g CLSI -o FEP_CLSI -gc 2
echo "Completed: FEP model saved to FEP_CLSI/"
echo ""

## Example 3: Causal inference analysis for Cefoxitin MIC values
## -gc 3: Use the third column from MIC file (Cefoxitin MIC) as target variable
## -mtdC PC: Perform causal discovery using PC algorithm (instead of prediction)
## Purpose: Identify causal relationships between ARGs and Cefoxitin minimum inhibitory concentration (MIC)
##          Outputs causal graph and edges between genes and MIC values
echo "Example 3: Causal inference for Cefoxitin MIC values (using PC algorithm)..."
python ../ModelFlow.py -i ARG.tsv -g MIC -o Cefoxitin_pc -gc 3 -mtdC PC
echo "Completed: Causal analysis results saved to Cefoxitin_pc/"
echo ""

echo "=========================================="
echo "All examples completed successfully!"
echo "Check the output directories for results:"
echo "- AMK_CLSI/     : AMK susceptibility prediction results"
echo "- FEP_CLSI/     : FEP susceptibility prediction results" 
echo "- Cefoxitin_pc/ : Cefoxitin MIC causal inference results"
echo "=========================================="