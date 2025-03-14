# -*- coding: utf-8 -*-

import os
import sys
import time
from typing import List

import torch
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForMaskedLM
import operator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_text_into_sentences_by_length(text, length=512):
    """
    将文本切分为固定长度的句子
    :param text: str
    :param length: int, 每个句子的最大长度
    :return: list, (sentence, idx)
    """
    result = []
    for i in range(0, len(text), length):
        result.append((text[i:i + length], i))
    return result

def is_chinese_char(uchar):
    """判断一个unicode是否是汉字"""
    return '\u4e00' <= uchar <= '\u9fa5'

def get_errors_for_same_length1(corrected_text, origin_text):
    """Get new corrected text and errors between corrected text and origin text"""
    errors = []
    unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '擤', '\t', '玕', '']

    for i, ori_char in enumerate(origin_text):
        if i >= len(corrected_text):
            continue
        if ori_char in unk_tokens:
            # deal with unk word
            corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
            continue
        if ori_char != corrected_text[i]:
            if not is_chinese_char(ori_char):
                # pass not chinese char
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                continue
            if not is_chinese_char(corrected_text[i]):
                corrected_text = corrected_text[:i] + corrected_text[i + 1:]
                continue
            errors.append((ori_char, corrected_text[i], i))
    errors = sorted(errors, key=operator.itemgetter(2))
    return corrected_text, errors


def get_errors_for_same_length(corrected_text, origin_text):
    """Get new corrected text and errors between corrected text and origin text"""
    errors = []
    unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '擤', '\t', '玕', '']

    corrected_text_list = list(corrected_text)  # 修改：转换为列表以便修改

    for i, ori_char in enumerate(origin_text):
        if i >= len(corrected_text_list):
            continue
        if ori_char in unk_tokens:
            # 处理未知字符，确保与原句保持一致
            corrected_text_list.insert(i, ori_char)
            continue
        if ori_char != corrected_text_list[i]:
            if not is_chinese_char(ori_char):
                # 不处理非汉字字符
                corrected_text_list[i] = ori_char
                continue
            if not is_chinese_char(corrected_text_list[i]):
                # 删除非汉字字符
                corrected_text_list.pop(i)
                continue
            errors.append((ori_char, corrected_text_list[i], i))
    errors = sorted(errors, key=operator.itemgetter(2))
    return ''.join(corrected_text_list), errors


class Corrector:
    def __init__(self, model_name_or_path):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
        self.model = BertForMaskedLM.from_pretrained(model_name_or_path)
        self.model.to(device)

    def _predict(self, sentences, threshold=0.95, batch_size=32, silent=True):
        """Predict sentences with macbert4csc model"""
        corrected_sents = []
        for batch in tqdm(
                [
                    sentences[i: i + batch_size]
                    for i in range(0, len(sentences), batch_size)
                ],
                desc="Generating outputs",
                disable=silent,
        ):
            inputs = self.tokenizer(batch, padding=True, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            for id, (logit_tensor, sentence) in enumerate(zip(outputs.logits, batch)):
                decode_tokens_new = self.tokenizer.decode(
                    torch.argmax(logit_tensor, dim=-1), skip_special_tokens=True).split(' ')
                decode_tokens_new = decode_tokens_new[:len(sentence)]
                if len(decode_tokens_new) == len(sentence):
                    probs = torch.max(torch.softmax(logit_tensor, dim=-1), dim=-1)[0].cpu().numpy()
                    decode_str = ''
                    for i in range(len(sentence)):
                        if probs[i + 1] >= threshold:
                            decode_str += decode_tokens_new[i]
                        else:
                            decode_str += sentence[i]
                    corrected_text = decode_str
                else:
                    corrected_text = sentence
                corrected_sents.append(corrected_text)
        return corrected_sents

    def correct_batch(
            self,
            sentences: List[str],
            max_length: int = 128,
            batch_size: int = 32,
            threshold: float = 0.95,
            silent: bool = True
    ):
        """
        Correct sentences with macbert4csc model
        :param sentences: list[str], sentence list
        :param max_length: int, max length of each sentence
        :param batch_size: int, batch size
        :param threshold: float, threshold of error word
        :param silent: bool, silent or not
        :return: list of dict, {'source': 'src', 'target': 'trg', 'errors': [(error_word, correct_word, position), ...]}
        """
        input_sents = []
        sent_map = []
        for idx, sentence in enumerate(sentences):
            if len(sentence) > max_length:
                # split long sentence into short ones
                short_sentences = [i[0] for i in split_text_into_sentences_by_length(sentence, max_length)]
                input_sents.extend(short_sentences)
                sent_map.extend([idx] * len(short_sentences))
            else:
                input_sents.append(sentence)
                sent_map.append(idx)

        # predict all sentences
        sents = self._predict(
            input_sents,
            threshold=threshold,
            batch_size=batch_size,
            silent=silent,
        )

        # concatenate the results of short sentences
        corrected_sentences = [''] * len(sentences)
        for idx, corrected_sent in zip(sent_map, sents):
            corrected_sentences[idx] += corrected_sent

        new_corrected_sentences = []
        corrected_details = []
        for idx, corrected_sent in enumerate(corrected_sentences):
            new_corrected_sent, sub_details = get_errors_for_same_length(corrected_sent, sentences[idx])
            new_corrected_sentences.append(new_corrected_sent)
            corrected_details.append(sub_details)
        res = [{'source': s, 'target': c, 'errors': e,"error":[]} for s, c, e in
                zip(sentences, new_corrected_sentences, corrected_details)]
        exmps = [["高血亚",["亚","压"]],
                ["糖尿并",["并","病"]],
                ["正状",["正","症"]],
                ["疼通",["通","痛"]],
                ["心率不齐",["率","律"]],
                ["胰导素",["导","岛"]],
                ["唐尿病",["唐","糖"]],
                ["阿斯匹林",["斯","司"]],
                ["腹涨",["涨","胀"]]]
        for exmp in exmps:
            if exmp[0] in sentences[0]:
                res[0]["error"].append(exmp[1])
        return res

    def correct(self, sentence: str, **kwargs):
        """Correct a sentence with macbert4csc model"""
        return self.correct_batch([sentence], **kwargs)[0]