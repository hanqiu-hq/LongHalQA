# Add the following functions to your existing utils.py file


def lhqa_doc_to_visual(doc):
    # Assuming the 'doc' dictionary has a key 'image' with image data
    return [doc["image"].convert("RGB")]


def lhqa_doc_to_text_binary(doc):
    # Assuming the 'doc' dictionary has a key 'question' with the question text
    question = doc["question"].strip()
    return "{}\nAnswer the question using a single word \'Yes\' or \'No\'.".format(question)


def lhqa_doc_to_text_choice(doc):
    question = doc["question"].strip()
    question += "\n" + f"A. {doc['choice_a']}\n"
    question += f"B. {doc['choice_b']}\n"
    question += f"C. {doc['choice_c']}\n"
    question += f"D. {doc['choice_d']}"
    return f"{question}\nAnswer with the option's letter from the given choices directly."


def lhqa_process_results(doc, results):
    ori_pred = results[0].lower().strip()
    if ori_pred.startswith("yes"):
        pred = "yes"
    elif ori_pred.startswith("no"):
        pred = "no"
    else:
        pred = "wrong response"
    gt_ans = doc["answer"].lower().strip()
    assert gt_ans in ["yes", "no"]
    score = 1.0 if pred == gt_ans else 0.0
    return {
        "original_prediction": ori_pred,
        "lhqa_accuracy": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans,},
        "lhqa_precision": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans,},
        "lhqa_recall": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans,},
        "lhqa_f1_score": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans,},
        "lhqa_yes_ratio": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans,},
    }


def lhqa_aggregate_accuracy(results):
    total_score = 0
    for result in results:
        total_score += result["score"]
    avg_score = total_score / len(results)
    return avg_score


def lhqa_aggregate_precision(results):
    true_positives = 0
    false_positives = 0
    for result in results:
        pred = result["prediction"]
        gt = result["ground_truth"]
        if gt == "yes" and pred == "yes":
            true_positives += 1
        elif gt == "no" and pred == "yes":
            false_positives += 1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    return precision


def lhqa_aggregate_recall(results):
    true_positives = 0
    false_negatives = 0
    for result in results:
        pred = result["prediction"]
        gt = result["ground_truth"]
        if gt == "yes" and pred == "yes":
            true_positives += 1
        elif gt == "yes" and pred == "no":
            false_negatives += 1
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall


def lhqa_aggregate_f1_score(results):
    precision = lhqa_aggregate_precision(results)
    recall = lhqa_aggregate_recall(results)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


def lhqa_aggregate_yes_ratio(results):
    yes_count = 0
    no_count = 0
    for result in results:
        pred = result["prediction"]
        if pred == "yes":
            yes_count += 1
        elif pred == "no":
            no_count += 1
    yes_ratio = yes_count / len(results)
    return yes_ratio


def lhqa_process_result_choice(doc, result):
    pred = result[0].strip()
    if len(pred) > 1:
        pred = pred[0]
    answer = doc["answer"]

    return {"lhqa_acc": {"pred": pred, "answer": answer, "question_id": doc["question_id"]}}


def lhqa_aggregate_choice_acc(results):
    total_count = 0
    total_correct = 0
    for result in results:
        if result["pred"] == result["answer"]:
            total_correct += 1
        total_count += 1
    accuracy = total_correct / total_count
    return accuracy


def lhqa_doc_to_choice(doc):
    return [doc["choice_a"], doc["choice_b"], doc["choice_c"], doc["choice_d"]]


def lhqa_doc_to_mc_target(doc):
    answer2choice = {"A": "choice_a", "B": "choice_b", "C": "choice_c", "D": "choice_d"}
    return doc[answer2choice[doc["answer"]]]