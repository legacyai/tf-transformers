import pandas as pd
import os


def score_glue_joint(eval_folder):
    """Join GLUE evaluation"""

    # Infer num_layers from a saved file MNLI
    df_sample = pd.read_csv(os.path.join(eval_folder, "mnli_eval.csv"))
    num_layers = df_sample.shape[1]
    
    # MRPC
    df = pd.read_csv(os.path.join(eval_folder, "mrpc_eval.csv"))
    mrpc_scores = []
    for i in range(num_layers):
        # Best per ;ayer
        score = ((df['accuracy_score_layer_{}'.format(i+1)] + + df['f1_score_layer_{}'.format(i+1)])/2.0).max()
        mrpc_scores.append(score)
    
    # MNLI
    df = pd.read_csv(os.path.join(eval_folder, "mnli_eval.csv"))
    df_mismatched = pd.read_csv(os.path.join(eval_folder, "mnli_eval_mismatched.csv"))

    mnli_scores = []
    for i in range(num_layers):
        score = ((df['accuracy_score_layer_{}'.format(i+1)] + + df_mismatched['accuracy_score_layer_{}'.format(i+1)])/2.0).max()
        mnli_scores.append(score)
        
    # QQP
    df = pd.read_csv(os.path.join(eval_folder, "qqp_eval.csv"))
    qqp_scores = []
    for i in range(num_layers):
        score = ((df['accuracy_score_layer_{}'.format(i+1)] + + df['f1_score_layer_{}'.format(i+1)])/2.0).max()
        qqp_scores.append(score)
        
    # STSB
    df = pd.read_csv(os.path.join(eval_folder, "stsb_eval.csv"))
    stsb_scores = []
    for i in range(num_layers):
        score = ((df['pearsonr_layer_{}'.format(i+1)] + + df['spearmanr_layer_{}'.format(i+1)])/2.0).max()
        stsb_scores.append(score)
        
    # COLA
    df = pd.read_csv(os.path.join(eval_folder, "cola_eval.csv"))
    cola_scores = []
    for i in range(num_layers):
        cola_scores.append(df['matthews_corrcoef_layer_{}'.format(i+1)].max())

    # QNLI 
    df = pd.read_csv(os.path.join(eval_folder, "qnli_eval.csv"))
    qnli_scores = []
    for i in range(num_layers):
        qnli_scores.append(df['accuracy_score_layer_{}'.format(i+1)].max())

    # RTE 
    df = pd.read_csv(os.path.join(eval_folder, "rte_eval.csv"))
    rte_scores = []
    for i in range(num_layers):
        rte_scores.append(df['accuracy_score_layer_{}'.format(i+1)].max())
        
    # SST2
    df = pd.read_csv(os.path.join(eval_folder, "sst2_eval.csv"))
    sst2_scores = []
    for i in range(num_layers):
        sst2_scores.append(df['accuracy_score_layer_{}'.format(i+1)].max())
        
    # Average
    df_results = pd.DataFrame([cola_scores, mnli_scores, mrpc_scores, qnli_scores, qqp_scores, rte_scores, sst2_scores, stsb_scores])
    df_results.index = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb']
    df_results.columns = ['layer_{}'.format(i+1) for i in range(num_layers)]
    df_results = df_results.transpose()
    df_results['glue_score'] = df_results.mean(axis=1)
    
    df_results.to_csv(os.path.join(eval_folder, "glue_results_joint.csv"), index=False)

    print("GLUE SCORE calculated")
    print("------------------------")
    print(df_results.to_markdown())
    
def score_glue(eval_folder):
    """Join GLUE evaluation"""

    # MRPC
    mrpc_scores = []
    df = pd.read_csv(os.path.join(eval_folder, "mrpc_eval.csv"))
    score = ((df['accuracy_score'] + df['f1_score'])/2.0).max()
    mrpc_scores.append(score)
    
    # MNLI
    df = pd.read_csv(os.path.join(eval_folder, "mnli_eval.csv"))
    df_mismatched = pd.read_csv(os.path.join(eval_folder, "mnli_eval_mismatched.csv"))

    mnli_scores = []
    score = ((df['accuracy_score'] + df_mismatched['accuracy_score'])/2.0).max()
    mnli_scores.append(score)
        
    # QQP
    df = pd.read_csv(os.path.join(eval_folder, "qqp_eval.csv"))
    qqp_scores = []
    score =score = ((df['accuracy_score'] + df['f1_score'])/2.0).max()
    qqp_scores.append(score)
            
    # STSB
    df = pd.read_csv(os.path.join(eval_folder, "stsb_eval.csv"))
    stsb_scores = []
    score = ((df['pearsonr'] + df['spearmanr'])/2.0).max()
    stsb_scores.append(score)
        
    # COLA
    df = pd.read_csv(os.path.join(eval_folder, "cola_eval.csv"))
    cola_scores = []
    cola_scores.append(df['matthews_corrcoef'].max())

    # QNLI 
    df = pd.read_csv(os.path.join(eval_folder, "qnli_eval.csv"))
    qnli_scores = []
    qnli_scores.append(df['accuracy_score'].max())

    # RTE 
    df = pd.read_csv(os.path.join(eval_folder, "rte_eval.csv"))
    rte_scores = []
    rte_scores.append(df['accuracy_score'].max())
            
    # SST2
    df = pd.read_csv(os.path.join(eval_folder, "sst2_eval.csv"))
    sst2_scores = []
    sst2_scores.append(df['accuracy_score'].max())
        
    # Average
    df_results = pd.DataFrame([cola_scores, mnli_scores, mrpc_scores, qnli_scores, qqp_scores, rte_scores, sst2_scores, stsb_scores])
    df_results.index = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb']
    df_results = df_results.transpose()
    df_results['glue_score'] = df_results.mean(axis=1)
    
    df_results.to_csv(os.path.join(eval_folder, "glue_results.csv"), index=False)

    print("GLUE SCORE calculated")
    print("------------------------")
    print(df_results.to_markdown())

def score(eval_folder, return_all_layer_outputs):
    """Score GLUE based on subtasks"""
    
    if return_all_layer_outputs:
        score_glue_joint(eval_folder)
    else:
        score_glue(eval_folder)