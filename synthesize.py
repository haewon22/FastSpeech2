import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 발음 사전
def read_lexicon(lex_path):
    # { 단어: 발음 } 을 저장할 딕셔너리
    lexicon = {}
    # lexicon 경로 열기
    with open(lex_path) as f:
        # 파일 한 줄씩 순회. line: 현재 줄
        for line in f:
            # \n을 제거하고, \s(공백문자) 기준으로 분리
            temp = re.split(r"\s+", line.strip("\n"))
            # 분리된 리스트의 단어 word에 저장
            word = temp[0]
            # word에 대한 발음을 phones에 저장
            phones = temp[1:]
            # 단어의 소문자 형태가 lexicon에 없다면
            if word.lower() not in lexicon:
                # 소문자를 key로, 그 발음을 value로 lexicon에 추가
                lexicon[word.lower()] = phones
    return lexicon

# 영어 데이터 전처리
def preprocess_english(text, preprocess_config):
    # 오른쪽 끝의 구두점(,.) 제거
    text = text.rstrip(punctuation)
    # read_lexicon 함수로 lexicon 로드
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    # Grapheme-to-Phoneme, g2p 변환 인스턴스 생성
    # g2p: 텍스트(단어)를 음소로 변환
    g2p = G2p()
    phones = []
    # text를 정규화 표현식에 있는 문자를 기준으로 분리
    words = re.split(r"([,;.\-\?\!\s+])", text)
    # 분리된 단어들 순회
    for w in words:
        # 현재 단어 w의 소문자 형태가 lexicon에 있다면
        if w.lower() in lexicon:
            # phones에 w의 발음 추가
            phones += lexicon[w.lower()]
        # 현재 단어 w의 소문자 형태가 lexicon에 없다면
        else:
            # g2p로 단어를 음소로 변환하고, 공백(" ")을 제거해서 리스트에 추가
            # 처리된 음소를 phones에 추가
            # ['a', 'bc', '.']
            phones += list(filter(lambda p: p != " ", g2p(w)))
    # 음소를 {}로 구분 - {a}{bc}{.}
    phones = "{" + "}{".join(phones) + "}"
    # {}로 둘러싸인 단어(\w: 문자나 숫자)가 아닌 텍스트를 {sp}로 대체 - {a}{bc}{sp}
    # {sp}: silence, pause
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    # }{를 다시 공백으로 대체 - {a bc sp}
    phones = phones.replace("}{", " ")

    # 원본(text), 변환된 음소 시퀀스 (phones) print
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    # text_to_sequence: Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

# 만다린어 데이터 전처리
def preprocess_mandarin(text, preprocess_config):
    # read_lexicon 함수로 lexicon 로드
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    # pinyins: 중국어의 공식 로마자 음성(발음) 표기법
    # pintin(): 입력된 중국어 텍스트를 pinyin 시퀀스로 변환
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    # 각 pinyin 순회
    for p in pinyins:
        # lexicon에 해당 pinyin 발음이 있다면 phones에 발음 추가
        if p in lexicon:
            phones += lexicon[p]
        # lexicon에 해당 pinyin 발음이 없다면, phones에 "sp" 추가
        else:
            phones.append("sp")

    # { a b sp c}
    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    # text_to_sequence: Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

# 음성 합성
def synthesize(model, step, configs, vocoder, batchs, control_values):
    # 각 config 저장
    preprocess_config, model_config, train_config = configs
    # 각 control(각 피처를 세부 조정할 값) 값들 저장
    pitch_control, energy_control, duration_control = control_values

    # 배치 처리
    for batch in batchs:
        batch = to_device(batch, device)
        # 합성 시엔 gradient가 불필요하므로 비활성화
        with torch.no_grad():
            # Forward
            # 현재 배치에 대해서 음성 합성
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            # 합성된 output과 vocoder를 사용해서 최종 오디오 샘플 생성
            # result_path에 저장됨
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]
    # 각 control 값으로 피처 조정
    control_values = args.pitch_control, args.energy_control, args.duration_control

    # 음성 합성
    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)
