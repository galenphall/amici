import sqlite3
import os
import pandas as pd
import networkx as nx 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_missing_data_patterns(G):
    """
    Analyze missing data patterns to inform modeling decisions.
    """
    if G is None:
        return None
    
    # Extract nodes
    amicus_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'amicus']
    docket_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'docket']
    
    print(f"Total amicus groups: {len(amicus_nodes)}")
    print(f"Total dockets: {len(docket_nodes)}")
    
    # Create initial vote matrix to analyze missingness
    vote_data = []
    position_data = []  # Keep track of actual positions
    
    for amicus in amicus_nodes:
        vote_row = []
        position_row = []
        for docket in docket_nodes:
            if G.has_edge(amicus, docket):
                position = G[amicus][docket]['position']
                position_row.append(position)
                # For now, just mark as participated (1) or not (0)
                vote_row.append(1)
            else:
                position_row.append('missing')
                vote_row.append(0)
        vote_data.append(vote_row)
        position_data.append(position_row)
    
    # Convert to DataFrames
    participation_df = pd.DataFrame(vote_data, 
                                  index=[G.nodes[n]['name'] for n in amicus_nodes],
                                  columns=[f"{G.nodes[n]['year']}_{G.nodes[n]['number']}" for n in docket_nodes])
    
    position_df = pd.DataFrame(position_data,
                             index=[G.nodes[n]['name'] for n in amicus_nodes],
                             columns=[f"{G.nodes[n]['year']}_{G.nodes[n]['number']}" for n in docket_nodes])
    
    # Calculate missing data statistics
    total_cells = participation_df.shape[0] * participation_df.shape[1]
    observed_cells = participation_df.sum().sum()
    missing_rate = 1 - (observed_cells / total_cells)
    
    print(f"\n=== MISSING DATA ANALYSIS ===")
    print(f"Total possible group-case combinations: {total_cells:,}")
    print(f"Observed combinations: {observed_cells:,}")
    print(f"Missing rate: {missing_rate:.1%}")
    
    # Analyze participation patterns by group
    group_participation = participation_df.sum(axis=1)
    print(f"\n=== GROUP PARTICIPATION ===")
    print(f"Mean cases per group: {group_participation.mean():.1f}")
    print(f"Median cases per group: {group_participation.median():.1f}")
    print(f"Groups with >50 cases: {(group_participation > 50).sum()}")
    print(f"Groups with >20 cases: {(group_participation > 20).sum()}")
    print(f"Groups with >10 cases: {(group_participation > 10).sum()}")
    print(f"Groups with >5 cases: {(group_participation > 5).sum()}")
    
    # Analyze participation patterns by case
    case_participation = participation_df.sum(axis=0)
    print(f"\n=== CASE PARTICIPATION ===")
    print(f"Mean groups per case: {case_participation.mean():.1f}")
    print(f"Median groups per case: {case_participation.median():.1f}")
    print(f"Cases with >50 groups: {(case_participation > 50).sum()}")
    print(f"Cases with >20 groups: {(case_participation > 20).sum()}")
    print(f"Cases with >10 groups: {(case_participation > 10).sum()}")
    print(f"Cases with >5 groups: {(case_participation > 5).sum()}")
    
    # Analyze position types
    all_positions = []
    for row in position_data:
        all_positions.extend(row)
    
    position_counts = pd.Series(all_positions).value_counts()
    print(f"\n=== POSITION TYPES ===")
    for pos, count in position_counts.items():
        if pos != 'missing':
            pct = count / observed_cells * 100
            print(f"{pos}: {count:,} ({pct:.1f}% of observed)")
    
    return {
        'participation_df': participation_df,
        'position_df': position_df,
        'group_participation': group_participation,
        'case_participation': case_participation,
        'missing_rate': missing_rate,
        'position_counts': position_counts
    }

def prepare_data_with_missing_strategy(G, strategy='conservative', min_cases=3, min_groups=3):
    """
    Prepare data with different strategies for handling missing data and ambiguous positions.
    
    Strategies:
    - 'conservative': Only use clear petitioner/respondent positions
    - 'inclusive': Include other position types as neutral/weak signals
    - 'ordinal': Create ordinal scale for different position strengths
    """
    
    if G is None:
        return None
    
    amicus_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'amicus']
    docket_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'docket']
    
    # Get all unique positions to understand the data
    all_positions = set()
    for amicus in amicus_nodes:
        for docket in docket_nodes:
            if G.has_edge(amicus, docket):
                position = G[amicus][docket]['position']
                all_positions.add(position)
    
    print(f"Unique position types found: {sorted(all_positions)}")
    
    # Create vote matrix based on strategy
    vote_data = []
    
    for amicus in amicus_nodes:
        row = []
        for docket in docket_nodes:
            if G.has_edge(amicus, docket):
                position = G[amicus][docket]['position']
                
                if strategy == 'conservative':
                    # Only clear petitioner/respondent positions
                    if position == 'petitioner':
                        vote = 1
                    elif position == 'respondent':
                        vote = -1
                    else:
                        vote = 0  # Treat ambiguous as missing
                        
                elif strategy == 'inclusive':
                    # Include more position types
                    if position in ['petitioner', 'petitioner_support']:
                        vote = 1
                    elif position in ['respondent', 'respondent_support']:
                        vote = -1
                    elif position in ['neutral', 'neither', 'other']:
                        vote = 0  # Truly neutral
                    else:
                        vote = 0  # Unknown positions as missing
                        
                elif strategy == 'ordinal':
                    # Create ordinal scale (for ordIRT)
                    # 1 = strongly respondent, 2 = neutral/missing, 3 = strongly petitioner
                    if position in ['respondent']:
                        vote = 1
                    elif position in ['respondent_support', 'leaning_respondent']:
                        vote = 1  # Could be 1.5 if we had more categories
                    elif position in ['neutral', 'neither', 'other']:
                        vote = 2
                    elif position in ['petitioner_support', 'leaning_petitioner']:
                        vote = 3  # Could be 2.5 if we had more categories
                    elif position in ['petitioner']:
                        vote = 3
                    else:
                        vote = 0  # Missing data
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
            else:
                vote = 0  # No participation = missing
            row.append(vote)
        vote_data.append(row)
    
    # Convert to DataFrame
    vote_df = pd.DataFrame(vote_data, 
                          index=[G.nodes[n]['name'] for n in amicus_nodes],
                          columns=[f"{G.nodes[n]['year']}_{G.nodes[n]['number']}" for n in docket_nodes])
    
    print(f"\nUsing '{strategy}' strategy:")
    print(f"Initial matrix shape: {vote_df.shape}")
    
    # Calculate statistics for this strategy
    if strategy == 'ordinal':
        non_missing = (vote_df != 0).sum().sum()
        total_cells = vote_df.shape[0] * vote_df.shape[1]
        print(f"Non-missing cells: {non_missing:,} ({non_missing/total_cells:.1%})")
        
        # Show value distribution
        value_counts = vote_df.values.flatten()
        unique, counts = np.unique(value_counts, return_counts=True)
        for val, count in zip(unique, counts):
            pct = count / len(value_counts) * 100
            if val == 0:
                print(f"  Missing (0): {count:,} ({pct:.1f}%)")
            else:
                print(f"  Category {val}: {count:,} ({pct:.1f}%)")
    else:
        # Binary analysis
        petitioner_votes = (vote_df == 1).sum().sum()
        respondent_votes = (vote_df == -1).sum().sum()
        missing_votes = (vote_df == 0).sum().sum()
        total_cells = vote_df.shape[0] * vote_df.shape[1]
        
        print(f"Petitioner positions: {petitioner_votes:,} ({petitioner_votes/total_cells:.1%})")
        print(f"Respondent positions: {respondent_votes:,} ({respondent_votes/total_cells:.1%})")
        print(f"Missing/neutral: {missing_votes:,} ({missing_votes/total_cells:.1%})")
    
    # Apply filtering
    if strategy == 'ordinal':
        # For ordinal, keep groups/cases with some participation
        group_participation = (vote_df != 0).sum(axis=1)
        case_participation = (vote_df != 0).sum(axis=0)
    else:
        # For binary, keep groups/cases with some clear positions
        group_participation = ((vote_df == 1) | (vote_df == -1)).sum(axis=1)
        case_participation = ((vote_df == 1) | (vote_df == -1)).sum(axis=0)
    
    active_groups = group_participation[group_participation >= min_cases].index
    active_cases = case_participation[case_participation >= min_groups].index
    
    vote_df_filtered = vote_df.loc[active_groups, active_cases]
    
    print(f"After filtering (min {min_cases} cases, min {min_groups} groups): {vote_df_filtered.shape}")
    
    # Final statistics
    if strategy == 'ordinal':
        final_non_missing = (vote_df_filtered != 0).sum().sum()
        final_total = vote_df_filtered.shape[0] * vote_df_filtered.shape[1]
        print(f"Final density: {final_non_missing/final_total:.1%}")
    else:
        final_clear_positions = ((vote_df_filtered == 1) | (vote_df_filtered == -1)).sum().sum()
        final_total = vote_df_filtered.shape[0] * vote_df_filtered.shape[1]
        print(f"Final density (clear positions): {final_clear_positions/final_total:.1%}")
    
    return {
        'vote_matrix': vote_df_filtered.values,
        'group_names': vote_df_filtered.index.tolist(),
        'case_names': vote_df_filtered.columns.tolist(),
        'n_groups': len(vote_df_filtered.index),
        'n_cases': len(vote_df_filtered.columns),
        'strategy': strategy,
        'raw_matrix': vote_df  # Keep unfiltered for analysis
    }

def recommend_model_approach(missing_analysis, data_strategies):
    """
    Recommend which IRT approach to use based on data characteristics.
    """
    print(f"\n=== MODEL RECOMMENDATION ===")
    
    missing_rate = missing_analysis['missing_rate']
    print(f"Overall missing rate: {missing_rate:.1%}")
    
    # Analyze different strategies
    for strategy, data in data_strategies.items():
        if data is None:
            continue
        print(f"\n{strategy.upper()} STRATEGY:")
        print(f"  Final matrix size: {data['n_groups']} groups × {data['n_cases']} cases")
        
        if strategy == 'ordinal':
            non_missing = (data['vote_matrix'] != 0).sum()
            total = data['vote_matrix'].size
            density = non_missing / total
            print(f"  Data density: {density:.1%}")
        else:
            clear_positions = ((data['vote_matrix'] == 1) | (data['vote_matrix'] == -1)).sum()
            total = data['vote_matrix'].size
            density = clear_positions / total
            print(f"  Clear position density: {density:.1%}")
    
    # Make recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    
    if missing_rate > 0.95:
        print("⚠️  VERY SPARSE DATA (>95% missing)")
        print("   Consider: Network IRT or different data aggregation strategy")
    elif missing_rate > 0.90:
        print("⚠️  SPARSE DATA (>90% missing)")
        print("   Recommended: Binary IRT with conservative strategy")
        print("   - Focus on clear petitioner/respondent positions only")
        print("   - Use higher minimum participation thresholds")
    elif missing_rate > 0.80:
        print("📊 MODERATELY SPARSE DATA (80-90% missing)")
        print("   Recommended: Binary IRT with inclusive strategy")
        print("   - Include more position types as signals")
        print("   - Consider bootstrapping for uncertainty quantification")
    else:
        print("✅ REASONABLE DATA DENSITY (<80% missing)")
        print("   Multiple options available:")
        print("   - Binary IRT for clear liberal/conservative positions")
        print("   - Ordinal IRT if you have meaningful position gradations")
        print("   - Dynamic IRT if you want to track changes over time")
    
    # Specific recommendations based on position types
    position_counts = missing_analysis['position_counts']
    clear_positions = position_counts.get('petitioner', 0) + position_counts.get('respondent', 0)
    total_observed = sum(v for k, v in position_counts.items() if k != 'missing')
    
    if clear_positions / total_observed > 0.8:
        print("\n✅ CLEAR POSITION DOMINANCE")
        print("   Most positions are clear petitioner/respondent")
        print("   → Binary IRT is well-suited")
    else:
        print("\n📊 MIXED POSITION TYPES")
        print("   Many ambiguous or neutral positions")
        print("   → Consider ordinal IRT or more sophisticated missing data handling")

def main():
    """
    Run complete missing data analysis and model recommendation.
    """
    print("=== AMICUS BRIEF MISSING DATA ANALYSIS ===\n")
    
    # Load graph (using your existing function)
    from your_original_script import amicus_graph  # Import your function
    G = amicus_graph()
    
    if G is None:
        print("Could not load data")
        return
    
    # Analyze missing data patterns
    missing_analysis = analyze_missing_data_patterns(G)
    
    # Test different data preparation strategies
    strategies = ['conservative', 'inclusive', 'ordinal']
    data_strategies = {}
    
    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"TESTING {strategy.upper()} STRATEGY")
        print(f"{'='*50}")
        
        try:
            data_strategies[strategy] = prepare_data_with_missing_strategy(
                G, strategy=strategy, min_cases=3, min_groups=3
            )
        except Exception as e:
            print(f"Error with {strategy} strategy: {e}")
            data_strategies[strategy] = None
    
    # Make recommendations
    recommend_model_approach(missing_analysis, data_strategies)
    
    return missing_analysis, data_strategies

if __name__ == "__main__":
    missing_analysis, data_strategies = main()