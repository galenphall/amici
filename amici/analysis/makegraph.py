import sqlite3
import os
import pandas as pd
import networkx as nx 
import numpy as np

def amicus_graph():
    # Get relative path to the database file
    # Get absolute path to the database file relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "..", "database", "supreme_court_docs.db")
    db_path = os.path.abspath(db_path)  # Resolve any relative path components

    # Check if the file exists
    if not os.path.exists(db_path):
        print(f"Database file not found at: {db_path}")
        print(f"Current working directory: {os.getcwd()}")
        return None
    else:
        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        print(f"Successfully connected to SQLite database at {db_path}")

    G = nx.Graph()

    # Query to get amici-docket relationships with positions
    cursor.execute("""
        SELECT 
            a.merged_name,
            dk.year,
            dk.number,
            dk.position
        FROM amici a
        JOIN documents d ON a.document_id = d.document_id
        JOIN dockets dk ON d.document_id = dk.document_id
        WHERE a.merged_name IS NOT NULL AND a.category = 'organization'
    """)

    results = cursor.fetchall()

    # Add nodes and edges to the bipartite graph
    for merged_name, year, number, position in results:
        amicus_node = f"amicus_{merged_name}"
        docket_node = f"docket_{year}_{number}"
        
        # Add nodes with bipartite attribute
        G.add_node(amicus_node, bipartite=0, type='amicus', name=merged_name)
        G.add_node(docket_node, bipartite=1, type='docket', year=year, number=number)
        
        # Add edge with position as metadata
        G.add_edge(amicus_node, docket_node, position=position)

    print(f"Created bipartite graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    conn.close()
    return G

def prepare_emirt_data(G, min_cases=3, min_groups=3):
    """
    Prepare data.
    
    Parameters:
    G: NetworkX bipartite graph
    min_cases: Minimum number of cases a group must participate in
    min_groups: Minimum number of groups that must participate in a case
    
    Returns:
    dict: Contains vote matrix, group names, case names, and metadata
    """
    
    if G is None:
        print("No graph provided")
        return None

    # Contruct the (min_cases, min_groups)-core graph
    # Filter to k-core: repeatedly remove nodes that don't meet minimum degree requirements
    if min_cases == min_groups:
        G = nx.k_core(G, min_cases)
    else:
        while True:
            removed_nodes = []
            
            # Remove amicus nodes with fewer than min_cases edges
            for node in list(G.nodes()):
                if G.nodes[node]['type'] == 'amicus' and G.degree(node) < min_cases:
                    removed_nodes.append(node)
            
            # Remove docket nodes with fewer than min_groups edges
            for node in list(G.nodes()):
                if G.nodes[node]['type'] == 'docket' and G.degree(node) < min_groups:
                    removed_nodes.append(node)
            
            # If no nodes to remove, we've found the k-core
            if not removed_nodes:
                break
            
            # Remove the nodes
            G.remove_nodes_from(removed_nodes)

    print(f"After k-core filtering: {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    amicus_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'amicus']
    docket_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'docket']
    
    print(f"Found {len(amicus_nodes)} amicus groups and {len(docket_nodes)} dockets")
    
    # Get all unique positions to understand the data
    all_positions = set()
    for amicus in amicus_nodes:
        for docket in docket_nodes:
            if G.has_edge(amicus, docket):
                position = G[amicus][docket]['position']
                all_positions.add(position)
    
    print(f"Unique position types found: {sorted(all_positions)}")
    
    # Create vote matrix 
    vote_data = []
    
    for amicus in amicus_nodes:
        row = []
        for docket in docket_nodes:
            if G.has_edge(amicus, docket):
                position = G[amicus][docket]['position']
                
                if position == 'P':
                    vote = 1
                elif position == 'R':
                    vote = -1
                else:
                    vote = 0  # Treat ambiguous as missing

            else:
                vote = 0  # No participation = missing
            row.append(vote)
        vote_data.append(row)
    
    # Convert to DataFrame
    vote_df = pd.DataFrame(vote_data, 
                          index=[G.nodes[n]['name'] for n in amicus_nodes],
                          columns=[f"{G.nodes[n]['year']}_{G.nodes[n]['number']}" for n in docket_nodes])
    
    print(f"Initial matrix shape: {vote_df.shape}")
    
    petitioner_votes = (vote_df == 1).sum().sum()
    respondent_votes = (vote_df == -1).sum().sum()
    missing_votes = (vote_df == 0).sum().sum()
    total_cells = vote_df.shape[0] * vote_df.shape[1]
    
    print(f"Petitioner positions: {petitioner_votes:,} ({petitioner_votes/total_cells:.1%})")
    print(f"Respondent positions: {respondent_votes:,} ({respondent_votes/total_cells:.1%})")
    print(f"Missing/neutral: {missing_votes:,} ({missing_votes/total_cells:.1%})")
    
    # For binary, keep groups/cases with some clear positions
    group_participation = ((vote_df == 1) | (vote_df == -1)).sum(axis=1)
    case_participation = ((vote_df == 1) | (vote_df == -1)).sum(axis=0)
    
    active_groups = group_participation[group_participation >= min_cases].index
    active_cases = case_participation[case_participation >= min_groups].index
    
    vote_df_filtered = vote_df.loc[active_groups, active_cases]
    
    print(f"After filtering (min {min_cases} cases, min {min_groups} groups): {vote_df_filtered.shape}")
    
    # Final statistics
    final_clear_positions = ((vote_df_filtered == 1) | (vote_df_filtered == -1)).sum().sum()
    final_total = vote_df_filtered.shape[0] * vote_df_filtered.shape[1]
    print(f"Final density (clear positions): {final_clear_positions/final_total:.1%}")
    
    # Extract metadata
    group_names = vote_df_filtered.index.tolist()
    case_names = vote_df_filtered.columns.tolist()
    
    # Create year information for cases
    case_years = []
    for case in case_names:
        year_str = case.split('_')[0]
        try:
            case_years.append(int(year_str))
        except ValueError:
            case_years.append(None)
    
    return {
        'vote_matrix': vote_df_filtered.values,  # NumPy array for R
        'group_names': group_names,
        'case_names': case_names,
        'case_years': case_years,
        'n_groups': len(group_names),
        'n_cases': len(case_names),
        'raw_matrix': vote_df  # Keep unfiltered for analysis
    }


def save_for_emirt(data_dict, output_dir="emirt_data"):
    """
    Save data in formats that R can easily load.
    """
    if data_dict is None:
        print("No data to save")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save vote matrix as CSV (R can read this easily)
    vote_df = pd.DataFrame(data_dict['vote_matrix'], 
                          index=data_dict['group_names'],
                          columns=data_dict['case_names'])
    vote_df.to_csv(f"{output_dir}/vote_matrix.csv")
    
    # Save metadata
    metadata = pd.DataFrame({
        'group_names': data_dict['group_names'],
        'group_id': range(len(data_dict['group_names']))
    })
    metadata.to_csv(f"{output_dir}/group_metadata.csv", index=False)
    
    case_metadata = pd.DataFrame({
        'case_names': data_dict['case_names'],
        'case_id': range(len(data_dict['case_names'])),
        'year': data_dict['case_years']
    })
    case_metadata.to_csv(f"{output_dir}/case_metadata.csv", index=False)
    
    # Save summary statistics
    summary = {
        'n_groups': data_dict['n_groups'],
        'n_cases': data_dict['n_cases'],
        'n_votes': int((data_dict['vote_matrix'] != 0).sum()),
        'density': float((data_dict['vote_matrix'] != 0).sum() / (data_dict['n_groups'] * data_dict['n_cases']))
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f"{output_dir}/summary_stats.csv", index=False)
    
    print(f"Data saved to {output_dir}/ directory:")
    print(f"  - vote_matrix.csv: {data_dict['n_groups']} x {data_dict['n_cases']} vote matrix")
    print(f"  - group_metadata.csv: Group names and IDs")
    print(f"  - case_metadata.csv: Case names, IDs, and years")
    print(f"  - summary_stats.csv: Summary statistics")

def main():
    """
    Main function to run the complete data preparation pipeline.
    """
    print("Creating bipartite graph from amicus brief data...")
    G = amicus_graph()
    
    if G is None:
        return
    
    print("\nPreparing data for emIRT analysis...")

    data_dict = prepare_emirt_data(G, min_cases=5, min_groups=5)

    save_for_emirt(data_dict)
    
    print("\nData preparation complete!")

if __name__ == "__main__":
    main()