# coding: utf-8
"""
This module holds various MT evaluation metrics.
"""

from external_metrics import Rouge, sacrebleu, mscoco_rouge
import numpy as np
import torch

WER_COST_DEL = 3
WER_COST_INS = 3
WER_COST_SUB = 4


def chrf(references, hypotheses):
    """
    Character F-score from sacrebleu
    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    return (
        sacrebleu.corpus_chrf(hypotheses=hypotheses, references=references).score * 100
    )


def bleu(references, hypotheses, level='word'):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)
    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    if level=='char':
        #split word
        references = [' '.join(list(r)) for r in references]
        hypotheses = [' '.join(list(r)) for r in hypotheses]
    bleu_scores = sacrebleu.raw_corpus_bleu(
        sys_stream=hypotheses, ref_streams=[references]
    ).scores
    scores = {}
    for n in range(len(bleu_scores)):
        scores["bleu" + str(n + 1)] = bleu_scores[n]
    return scores


def token_accuracy(references, hypotheses, level="word"):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.
    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :param level: segmentation level, either "word", "bpe", or "char"
    :return:
    """
    correct_tokens = 0
    all_tokens = 0
    split_char = " " if level in ["word", "bpe"] else ""
    assert len(hypotheses) == len(references)
    for hyp, ref in zip(hypotheses, references):
        all_tokens += len(hyp)
        for h_i, r_i in zip(hyp.split(split_char), ref.split(split_char)):
            # min(len(h), len(r)) tokens considered
            if h_i == r_i:
                correct_tokens += 1
    return (correct_tokens / all_tokens) * 100 if all_tokens > 0 else 0.0


def sequence_accuracy(references, hypotheses):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.
    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    assert len(hypotheses) == len(references)
    correct_sequences = sum(
        [1 for (hyp, ref) in zip(hypotheses, references) if hyp == ref]
    )
    return (correct_sequences / len(hypotheses)) * 100 if hypotheses else 0.0


# implement sltunet rouge eval
def rouge_deprecated(references, hypotheses, level='word'):
    #beta 1.2
    rouge_score = 0
    n_seq = len(hypotheses)
    if level=='char':
        #split word
        references = [' '.join(list(r)) for r in references]
        hypotheses = [' '.join(list(r)) for r in hypotheses]

    for h, r in zip(hypotheses, references):
        rouge_score += mscoco_rouge.calc_score(hypotheses=[h], references=[r]) / n_seq

    return rouge_score * 100

def rouge(references, hypotheses, level='word'):
    if level=='char':
        hyp = [list(x) for x in hypotheses]
        ref = [list(x) for x in references]
    else:
        hyp = [x.split() for x in hypotheses]
        ref = [x.split() for x in references]
    a = Rouge.rouge([' '.join(x) for x in hyp], [' '.join(x) for x in ref])
    return a['rouge_l/f_score']*100

def wer_list(references, hypotheses):
    total_error = total_del = total_ins = total_sub = total_ref_len = 0

    for r, h in zip(references, hypotheses):
        res = wer_single(r=r, h=h)
        total_error += res["num_err"]
        total_del += res["num_del"]
        total_ins += res["num_ins"]
        total_sub += res["num_sub"]
        total_ref_len += res["num_ref"]

    wer = (total_error / total_ref_len) * 100
    del_rate = (total_del / total_ref_len) * 100
    ins_rate = (total_ins / total_ref_len) * 100
    sub_rate = (total_sub / total_ref_len) * 100

    return {
        "wer": wer,
        "del_rate": del_rate,
        "ins_rate": ins_rate,
        "sub_rate": sub_rate,
        # "del":total_del,
        # "ins":total_ins,
        # "sub":total_sub,
        # "ref_len":total_ref_len,
        # "error":total_error,
    }


def wer_single(r, h):
    r = r.strip().split()
    h = h.strip().split()
    edit_distance_matrix = edit_distance(r=r, h=h)
    alignment, alignment_out = get_alignment(r=r, h=h, d=edit_distance_matrix)

    num_cor = np.sum([s == "C" for s in alignment])
    num_del = np.sum([s == "D" for s in alignment])
    num_ins = np.sum([s == "I" for s in alignment])
    num_sub = np.sum([s == "S" for s in alignment])
    num_err = num_del + num_ins + num_sub
    num_ref = len(r)

    return {
        "alignment": alignment,
        "alignment_out": alignment_out,
        "num_cor": num_cor,
        "num_del": num_del,
        "num_ins": num_ins,
        "num_sub": num_sub,
        "num_err": num_err,
        "num_ref": num_ref,
    }


def edit_distance(r, h):
    """
    Original Code from https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    """
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape(
        (len(r) + 1, len(h) + 1)
    )
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                # d[0][j] = j
                d[0][j] = j * WER_COST_INS
            elif j == 0:
                d[i][0] = i * WER_COST_DEL
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + WER_COST_SUB
                insert = d[i][j - 1] + WER_COST_INS
                delete = d[i - 1][j] + WER_COST_DEL
                d[i][j] = min(substitute, insert, delete)
    return d


def get_alignment(r, h, d):
    """
    Original Code from https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    This function is to get the list of steps in the process of dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calculating the editing distance of h and r.
    """
    x = len(r)
    y = len(h)
    max_len = 3 * (x + y)

    alignlist = []
    align_ref = ""
    align_hyp = ""
    alignment = ""

    while True:
        if (x <= 0 and y <= 0) or (len(alignlist) > max_len):
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] and r[x - 1] == h[y - 1]:
            align_hyp = " " + h[y - 1] + align_hyp
            align_ref = " " + r[x - 1] + align_ref
            alignment = " " * (len(r[x - 1]) + 1) + alignment
            alignlist.append("C")
            x = max(x - 1, 0)
            y = max(y - 1, 0)
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] + WER_COST_SUB:
            ml = max(len(h[y - 1]), len(r[x - 1]))
            align_hyp = " " + h[y - 1].ljust(ml) + align_hyp
            align_ref = " " + r[x - 1].ljust(ml) + align_ref
            alignment = " " + "S" + " " * (ml - 1) + alignment
            alignlist.append("S")
            x = max(x - 1, 0)
            y = max(y - 1, 0)
        elif y >= 1 and d[x][y] == d[x][y - 1] + WER_COST_INS:
            align_hyp = " " + h[y - 1] + align_hyp
            align_ref = " " + "*" * len(h[y - 1]) + align_ref
            alignment = " " + "I" + " " * (len(h[y - 1]) - 1) + alignment
            alignlist.append("I")
            x = max(x, 0)
            y = max(y - 1, 0)
        else:
            align_hyp = " " + "*" * len(r[x - 1]) + align_hyp
            align_ref = " " + r[x - 1] + align_ref
            alignment = " " + "D" + " " * (len(r[x - 1]) - 1) + alignment
            alignlist.append("D")
            x = max(x - 1, 0)
            y = max(y, 0)

    align_ref = align_ref[1:]
    align_hyp = align_hyp[1:]
    alignment = alignment[1:]

    return (
        alignlist[::-1],
        {"align_ref": align_ref, "align_hyp": align_hyp, "alignment": alignment},
    )

def sableu(references, hypotheses, tokenizer):
    """
    Sacrebleu (with tokenization)

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    bleu_scores = sacrebleu.corpus_bleu(
        sys_stream=hypotheses, ref_streams=[references], tokenize=tokenizer,
    ).scores
    scores = {}
    for n in range(len(bleu_scores)):
        scores["bleu" + str(n + 1)] = bleu_scores[n]

    return scores

def translation_performance(txt_ref, txt_hyp):
    from rouge import Rouge as SLT_Rouge
    rouge=SLT_Rouge()
    scores = rouge.get_scores(txt_hyp, txt_ref, avg=True)
    scores['rouge-l']['f'] = scores['rouge-l']['f']*100

    tokenizer_args = '13a'
    # print('Signature: BLEU+case.mixed+numrefs.1+smooth.exp+tok.%s+version.1.4.2' % tokenizer_args)
    sableu_dict = sableu(references=txt_ref, hypotheses=txt_hyp, tokenizer=tokenizer_args)
    # print('BLEU', sableu_dict)
    # print('Signature: chrF2+case.mixed+numchars.6+numrefs.1+space.False+version.1.4.2')
    # print('Chrf', chrf(references=txt_ref, hypotheses=txt_hyp))

    print(sableu_dict)
    print(f"Rough: {scores['rouge-l']['f']:.2f}")

    # res = []
    # for n in range(4):
    #     res.append(f"{sableu_dict['bleu' + str(n + 1)]:.2f}")
    # res.append(f"{scores['rouge-l']['f']:.2f}")

    # print(" & ".join(res))

    return sableu_dict, float(scores['rouge-l']['f'])

def islr_performance(txt_ref, txt_hyp):
    true_sample = 0
    for tgt_pre, tgt_ref in zip(txt_hyp, txt_ref):
        true_sample += (tgt_pre == tgt_ref)

    top1_acc_pi = true_sample / len(txt_hyp) * 100

    gt_dict = {}
    pred_dict = {}
    for tgt_pre, tgt_ref in zip(txt_hyp, txt_ref):
        if tgt_ref in gt_dict.keys():
            gt_dict[tgt_ref] += 1
            pred_dict[tgt_ref] += (tgt_pre == tgt_ref)
        else:
            gt_dict[tgt_ref] = 1
            pred_dict[tgt_ref] = (tgt_pre == tgt_ref)

    mean_acc_pc = []
    for key in gt_dict.keys():
        mean_acc_pc.append(pred_dict[key] / gt_dict[key])
    top1_acc_pc = np.array(mean_acc_pc).mean() * 100

    print(f"top1_acc_pi: {top1_acc_pi:.2f}")
    print(f"top1_acc_pc: {top1_acc_pc:.2f}")
    return top1_acc_pi, top1_acc_pc


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for handling class imbalance in ISLR tasks.

    Args:
        alpha (float): Weighting factor for the rare class, default is 0.25
        gamma (float): Focusing parameter, default is 2.0
        reduction (str): Reduction method, default is 'mean'
        label_smoothing (float): Label smoothing factor, default is 0.0
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', label_smoothing=0.0, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted logits, shape [B, C] where C is the number of classes
            targets (torch.Tensor): Ground truth class indices, shape [B]

        Returns:
            torch.Tensor: Computed loss
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0.0:
            # Create smoothed targets
            num_classes = inputs.size(-1)
            smoothed_targets = torch.zeros_like(inputs).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            smoothed_targets = smoothed_targets * (1.0 - self.label_smoothing) + \
                              self.label_smoothing / num_classes

            # Compute log probabilities
            log_probs = torch.nn.functional.log_softmax(inputs, dim=-1)
            loss = -(smoothed_targets * log_probs).sum(dim=-1)

            # Apply focal weighting
            probs = torch.exp(log_probs)
            focal_weight = (1 - probs) ** self.gamma
            loss = focal_weight * loss

            # Apply alpha weighting for positive samples
            alpha_weight = torch.ones_like(loss)
            alpha_weight[targets > 0] = self.alpha  # Apply alpha only to positive classes
            loss = alpha_weight * loss
        else:
            # Standard focal loss implementation
            ce_loss = torch.nn.functional.cross_entropy(
                inputs, targets, reduction='none', label_smoothing=0.0,
                ignore_index=self.ignore_index
            )
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** self.gamma

            # Apply alpha weighting
            alpha_weight = torch.ones_like(ce_loss)
            alpha_weight[targets > 0] = self.alpha  # Apply alpha only to positive classes
            loss = alpha_weight * focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def islr_performance_topk(logits, target_token_ids, ks=[1, 3, 5, 7, 10]):
    """
    Calculates both per-instance and per-class Top-K accuracy for ISLR.

    Args:
        logits (torch.Tensor): Logits from the model for the first prediction step.
                               Shape: (batch_size, vocab_size)
        target_token_ids (torch.Tensor): Ground truth token IDs for the target class label.
                                         Shape: (batch_size,)
        ks (list): List of integers for K values.

    Returns:
        dict: Dictionary mapping metric names to their values:
              - 'topk_acc_pi': per-instance accuracy for each K
              - 'topk_acc_pc': per-class accuracy for each K
    """
    max_k = max(ks)
    batch_size = logits.size(0)

    # Ensure inputs are on same device and correct type
    target_token_ids = target_token_ids.to(device=logits.device, dtype=torch.long)

    # Get top K predictions for each sample
    _, topk_indices = torch.topk(logits.float(), max_k, dim=-1)  # Shape: (batch_size, max_k)

    # Dictionary to store accuracies
    accuracies = {}

    # Per-instance calculation
    # Compare target with top K predictions
    target_expanded = target_token_ids.unsqueeze(1).expand_as(topk_indices[:, :max_k])
    correct_matrix = (topk_indices[:, :max_k] == target_expanded)  # Shape: (batch_size, max_k)

    for k in ks:
        # Check if target is in top K predictions for each sample
        correct_k = correct_matrix[:, :k].any(dim=1)  # Shape: (batch_size,)
        accuracy_k = (correct_k.float().sum() / batch_size) * 100
        accuracies[f'top{k}_acc_pi'] = accuracy_k.item()

    # Per-class calculation
    # Create dictionaries to track per-class statistics
    class_correct = {}  # {class_id: {k: correct_count}}
    class_total = {}   # {class_id: total_count}

    for i in range(batch_size):
        target = target_token_ids[i].item()
        if target not in class_total:
            class_total[target] = 0
            class_correct[target] = {k: 0 for k in ks}
        class_total[target] += 1

        # Update correct counts for each K
        for k in ks:
            if correct_matrix[i, :k].any():
                class_correct[target][k] += 1

    # Calculate per-class accuracies for each K
    for k in ks:
        class_accuracies = []
        for class_id in class_total:
            if class_total[class_id] > 0:  # Avoid division by zero
                class_acc = (class_correct[class_id][k] / class_total[class_id]) * 100
                class_accuracies.append(class_acc)

        # Average across all classes
        accuracies[f'top{k}_acc_pc'] = float(sum(class_accuracies)) / len(class_accuracies)

    # Print both per-instance and per-class accuracies
    print("\nTop-K Accuracies (Per Instance):", {k: accuracies[f'top{k}_acc_pi'] for k in ks})
    print("Top-K Accuracies (Per Class):", {k: accuracies[f'top{k}_acc_pc'] for k in ks})

    return accuracies
