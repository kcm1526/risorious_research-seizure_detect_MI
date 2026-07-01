# SeizureTransformer Mechanistic Interpretability 실험 보고서

## 1. 실험 목적

본 실험의 목적은 학습된 SeizureTransformer가 EEG seizure detection을 수행할 때, 단순히 성능이 좋은지를 확인하는 데서 그치지 않고 **모델이 어떤 입력 구간을 보고 판단하는지**, **그 구간을 제거하면 예측이 어떻게 바뀌는지**, 그리고 **모델 내부 latent representation 중 어떤 부분이 seizure 판단에 관여하는지**를 분석하는 것이다.

이를 위해 세 가지 방법을 비교했다.

1. **Attribution-only**
   - 모델을 그대로 둔 상태에서 input EEG에 대한 attribution score를 계산한다.
   - 즉, 모델이 어느 시간대와 어느 채널을 중요하게 사용했는지 보는 방법이다.

2. **Attribution-guided masking**
   - attribution score가 높은 입력 시간 구간을 찾은 뒤, 해당 EEG 입력을 masking한다.
   - 그 후 모델 예측이 어떻게 변하는지 확인한다.
   - 즉, “중요하다고 나온 입력 구간이 실제로 중요한가?”를 검증한다.

3. **MI-guided latent intervention**
   - 입력 EEG가 아니라 Transformer 이후의 내부 latent representation을 대상으로 attribution을 계산한다.
   - attribution이 높은 latent element를 직접 suppress한다.
   - 그 후 decoder를 통과시켜 예측 변화를 확인한다.
   - 즉, “모델 내부에서 어떤 latent feature가 seizure 판단을 만들고 있는가?”를 분석한다.

---

## 2. Attribution 계산 방식

이번 실험에서는 기본적으로 다음 attribution score를 사용했다.

```text
attribution = abs(gradient * activation)
```

입력 attribution의 경우:

```text
input attribution = abs(d objective / d input * input)
```

latent attribution의 경우:

```text
latent attribution = abs(d objective / d latent * latent)
```

여기서 objective는 기본적으로 `label_positive`이다. 즉, 실제 seizure label이 1인 time-step들에서의 모델 평균 예측값을 scalar objective로 잡고, 그 objective에 대한 gradient를 계산했다.

따라서 attribution score가 높다는 것은 다음을 의미한다.

> 해당 input 또는 latent 값이 seizure-positive objective에 민감하게 연결되어 있다.

다만 `abs(...)`를 사용했기 때문에 attribution heatmap은 **중요도의 크기**를 보여주지만, 해당 요소가 seizure score를 올리는 방향인지 내리는 방향인지는 직접 보여주지 않는다.

---

## 3. 세 방법의 구현 방식

### 3.1 Attribution-only

Attribution-only는 원본 모델의 prediction을 바꾸지 않는다.

실행 흐름은 다음과 같다.

```text
EEG input
→ trained SeizureTransformer
→ prediction
→ input gradient 계산
→ input attribution heatmap 생성
```

따라서 Attribution-only의 detection score는 사실상 원본 모델의 baseline 성능으로 해석할 수 있다. 이 방법은 intervention을 하지 않기 때문에 delta 값은 0이다.

---

### 3.2 Attribution-guided masking

Attribution-guided masking은 먼저 input attribution을 계산한 뒤, 중요한 입력 시간 구간을 선택한다.

기본 설정은 다음과 같다.

```text
input attribution: channel x time
time score = channel 방향 평균
상위 10% time-step 선택
선택된 time-step의 EEG 입력을 0으로 masking
다시 모델 예측 수행
```

즉, 이 방법은 다음 질문에 답하려고 한다.

> Attribution이 중요하다고 말한 EEG 구간을 실제로 제거하면 seizure prediction이 줄어드는가?

만약 masking 후 seizure prediction이 크게 떨어진다면, attribution이 가리킨 구간이 실제로 모델 판단에 중요한 구간이었다고 볼 수 있다.

---

### 3.3 MI-guided latent intervention

MI-guided latent intervention은 입력이 아니라 모델 내부 representation에 개입한다.

SeizureTransformer의 흐름을 단순화하면 다음과 같다.

```text
EEG input
→ CNN encoder
→ ResCNN stack
→ positional encoding
→ Transformer encoder
→ latent representation
→ decoder
→ seizure prediction
```

본 실험에서는 Transformer 이후 latent representation에 대해 attribution을 계산했다.

기본 설정은 다음과 같다.

```text
latent attribution = abs(gradient * latent)
상위 10% latent element 선택
선택된 latent element를 0으로 suppress
decoder를 다시 통과시켜 prediction 계산
```

즉, 이 방법은 다음 질문을 던진다.

> 모델 내부 latent 중 seizure 판단에 중요해 보이는 요소를 직접 꺼버리면 예측이 어떻게 바뀌는가?

이 방법은 입력 자체를 훼손하지 않고 모델 내부 표현을 조작한다는 점에서, input masking보다 더 mechanistic한 개입이라고 볼 수 있다.

---

## 4. Aggregate 결과 요약

공유된 summary 결과를 바탕으로 세 방법의 주요 성능을 정리하면 다음과 같다.

### 4.1 Sample-level 결과

| Method | F1 | Sensitivity | Precision | FP Rate |
|---|---:|---:|---:|---:|
| Attribution-only | 0.5003 | 0.3833 | 0.7201 | 763.86 |
| Attribution-guided masking | 0.4329 | 0.4003 | 0.4713 | 2302.31 |
| MI-guided latent intervention | 0.3505 | 0.2298 | 0.7391 | 415.83 |

### 4.2 Event-level 결과

| Method | F1 | Sensitivity | Precision | FP Rate |
|---|---:|---:|---:|---:|
| Attribution-only | 0.6346 | 0.6224 | 0.6472 | 21.66 |
| Attribution-guided masking | 0.5964 | 0.5929 | 0.6000 | 25.24 |
| MI-guided latent intervention | 0.5755 | 0.4720 | 0.7373 | 10.73 |

---

## 5. 결과 해석

### 5.1 Attribution-only 결과

Attribution-only는 모델에 intervention을 하지 않기 때문에 원본 모델 성능에 해당한다.

Event-level 기준으로 F1은 0.6346이며, sensitivity와 precision이 각각 0.6224, 0.6472로 비교적 균형 잡힌 결과를 보였다.

Attribution 관련 지표는 다음과 같다.

```text
attribution_label_auc = 0.7879
attribution_mass_on_seizure = 0.0931
label_positive_rate = 0.0421
normalized_attribution_entropy = 0.9125
```

label positive rate가 약 4.2%인데, attribution mass on seizure는 약 9.3%이다. 이는 attribution이 완전히 균일하게 퍼져 있는 것이 아니라 seizure label 구간에 상대적으로 더 많이 모여 있음을 의미한다.

즉, input attribution은 어느 정도 seizure 구간을 잘 가리키고 있다.

---

### 5.2 Attribution-guided masking 결과

Attribution-guided masking에서는 attribution이 높은 input time 구간을 masking한 뒤 다시 예측했다.

Event-level F1은 0.6346에서 0.5964로 감소했다. Precision도 0.6472에서 0.6000으로 낮아졌고, FP rate는 21.66에서 25.24로 증가했다.

Prediction delta는 다음과 같다.

```text
mean_delta = +0.0727
mean_delta_on_background = +0.0781
mean_delta_on_seizure = -0.0607
mean_abs_delta_on_seizure = 0.1723
```

중요한 점은 masking이 단순히 모든 prediction을 낮춘 것이 아니라는 점이다.

특히 seizure 구간에서는 평균 prediction이 감소했다.

```text
mean_delta_on_seizure = -0.0607
```

이는 attribution이 가리킨 구간이 seizure prediction에 실제로 중요했음을 보여준다.

하지만 background 구간에서는 평균 prediction이 오히려 증가했다.

```text
mean_delta_on_background = +0.0781
```

이는 input을 zero masking하면서 모델 입장에서는 자연스럽지 않은 입력 분포가 만들어졌고, 그 결과 일부 background 구간에서 예측이 불안정해졌을 가능성을 시사한다.

따라서 Attribution-guided masking은 다음과 같이 해석할 수 있다.

> Attribution이 찾은 입력 구간은 실제로 seizure prediction에 중요하다. 그러나 입력을 직접 가리는 방식은 distribution shift를 만들 수 있어, background false positive를 증가시킬 수도 있다.

---

### 5.3 MI-guided latent intervention 결과

MI-guided latent intervention에서는 input EEG가 아니라 Transformer latent representation의 중요 element를 suppress했다.

Event-level 결과를 보면 F1은 0.5755로 감소했지만, precision은 0.7373으로 가장 높았다. FP rate도 10.73으로 가장 낮았다.

반면 sensitivity는 0.4720으로 낮아졌다.

즉, latent intervention 이후 모델은 더 조심스럽게 seizure라고 판단하게 되었고, false positive는 줄었지만 실제 seizure event를 놓치는 경우가 늘어났다.

Prediction delta는 다음과 같다.

```text
mean_abs_delta = 0.0520
mean_abs_delta_on_background = 0.0489
mean_abs_delta_on_seizure = 0.1090
mean_delta_on_background = +0.0409
mean_delta_on_seizure = -0.0128
```

seizure 구간에서 prediction이 평균적으로 감소했다는 점은 latent intervention이 seizure-related internal representation을 약화시켰음을 의미한다.

Attribution 관련 지표는 매우 강하게 나타났다.

```text
attribution_label_auc = 0.9929
attribution_mass_on_seizure = 0.1328
label_positive_rate = 0.0421
normalized_attribution_entropy = 0.9718
```

특히 attribution_label_auc가 0.9929로 매우 높다. 이는 latent attribution이 seizure label과 매우 잘 정렬되어 있음을 의미한다.

또한 label positive rate는 4.2%인데 attribution mass on seizure는 13.3%이다. 즉, 모델 내부 latent attribution은 seizure 구간에 훨씬 더 집중되어 있다.

따라서 MI-guided latent intervention은 다음과 같이 해석할 수 있다.

> Transformer latent representation 안에는 seizure 판단과 강하게 연결된 내부 feature들이 존재한다. 이 feature들을 suppress하면 seizure detection sensitivity가 낮아지고, 모델은 더 보수적으로 변한다.

---

## 6. Example figure 해석

첨부된 example figure는 하나의 약 60초 EEG window에 대해 input-level attribution, input masking 효과, latent-level attribution, latent intervention 효과를 함께 보여준다.

### 6.1 Input EEG window

첫 번째 panel은 원본 EEG 입력이다.

x축은 시간이고, y축은 EEG channel이다. 색은 z-scored EEG amplitude를 나타낸다.

그림에서 약 20-30초, 그리고 35-40초 구간에 여러 channel에 걸친 강한 EEG 패턴이 보인다. 이 구간은 이후 attribution map에서도 강하게 나타난다.

---

### 6.2 Input attribution heatmap

두 번째 panel은 input attribution heatmap이다.

```text
input attribution = abs(gradient * input)
```

이 heatmap은 모델이 EEG의 어느 시간과 어느 채널에 민감하게 반응했는지를 보여준다.

밝은 부분은 모델이 seizure-positive objective를 계산할 때 중요하게 사용한 input 위치이다.

그림에서 20-30초, 35-40초 구간이 밝게 나타난다. 이는 모델이 해당 구간의 EEG 패턴을 seizure 판단에 중요하게 사용했다는 뜻이다.

---

### 6.3 Input attribution collapsed over channels

세 번째 panel은 input attribution을 channel 방향으로 평균낸 것이다.

즉, 복잡한 channel x time heatmap을 시간축 하나의 curve로 요약한 것이다.

```text
time score(t) = average over channels of input attribution(channel, t)
```

이 curve가 높은 시간대는 모델이 전체 EEG channel을 통틀어 중요하게 본 시간대이다.

연한 세로 음영은 실제로 masking에 선택된 input time 구간이다. 이 음영은 attribution score가 높은 time-step을 기준으로 선택되었다.

따라서 이 panel은 다음을 보여준다.

> 어떤 시간대가 input attribution 기준으로 중요했고, 그중 어떤 구간이 masking 대상으로 선택되었는가?

---

### 6.4 Prediction effect of the interventions

네 번째 panel은 가장 중요한 결과 해석 panel이다.

여기에는 세 가지 prediction curve가 들어 있다.

```text
Original prediction: 원본 모델 예측
After input masking: 중요한 input 구간을 가린 뒤 예측
After latent intervention: 중요한 latent element를 suppress한 뒤 예측
```

수평 dashed line은 seizure threshold이다. 붉은 영역 또는 선은 실제 seizure label이다.

그림을 보면 원본 prediction은 20-30초, 35-40초 부근에서 크게 상승한다. 그런데 input masking 이후 prediction은 해당 구간에서 크게 낮아진다.

이는 다음을 의미한다.

> Attribution이 중요하다고 판단한 EEG 구간을 실제로 가렸더니 seizure prediction이 약해졌다.

따라서 input attribution은 단순한 시각적 설명이 아니라, 모델 prediction과 causal하게 연결된 부분을 어느 정도 잡아냈다고 볼 수 있다.

Latent intervention curve는 input masking과는 다른 변화를 보인다. 이는 latent intervention이 입력 신호를 직접 제거하는 것이 아니라, 모델 내부 representation의 일부만 suppress하기 때문이다.

---

### 6.5 Transformer latent attribution heatmap

다섯 번째 panel은 Transformer latent representation에 대한 attribution heatmap이다.

입력 heatmap과 달리 y축은 EEG channel이 아니라 latent feature index이다.

즉, 이 panel은 다음을 보여준다.

> 모델 내부의 어떤 latent feature가 어느 시간대에서 seizure prediction에 중요했는가?

그림에서 latent attribution 역시 20-30초, 35-40초 부근에서 강하게 나타난다.

이는 input-level attribution과 latent-level attribution이 비슷한 시간 구간을 가리키고 있음을 의미한다.

즉, 모델은 입력 EEG의 중요한 구간을 내부 latent representation에서도 seizure-relevant feature로 변환하고 있다고 해석할 수 있다.

---

### 6.6 Latent attribution collapsed over features

마지막 panel은 latent attribution을 feature 방향으로 평균낸 것이다.

```text
latent time score(t) = average over latent features of latent attribution(feature, t)
```

이 curve는 모델 내부 representation 기준으로 어느 시간대가 중요한지를 보여준다.

연한 초록색 음영은 latent intervention이 적용된 시간 영역을 의미한다. 오른쪽 작은 bar plot은 attribution이 높았던 latent feature dimension들을 보여준다.

따라서 이 panel은 다음 질문에 답한다.

> 모델 내부에서 어떤 시간대와 어떤 latent feature가 suppress 대상이 되었는가?

---

## 7. 전체 결론

이번 실험을 통해 다음을 확인했다.

첫째, Attribution-only 결과에서 input attribution은 실제 seizure label 구간에 어느 정도 집중되어 있었다. 따라서 모델의 판단은 무작위적인 입력 위치가 아니라 seizure-relevant EEG 구간에 기반하고 있을 가능성이 있다.

둘째, Attribution-guided masking에서는 attribution이 높은 input 구간을 제거했을 때 seizure 구간의 prediction이 감소했다. 이는 input attribution이 실제 모델 판단에 중요한 구간을 잡고 있음을 보여준다. 다만 zero masking은 background prediction을 불안정하게 만들 수 있어 false positive가 증가하는 부작용도 관찰되었다.

셋째, MI-guided latent intervention에서는 Transformer latent attribution이 seizure label과 매우 강하게 정렬되었다. 중요한 latent element를 suppress하면 모델의 sensitivity가 낮아지고 precision이 높아졌다. 이는 모델 내부에 seizure 판단과 직접 연결된 latent representation이 존재함을 시사한다.

넷째, example figure에서는 input attribution이 강한 구간과 latent attribution이 강한 구간이 20-30초, 35-40초 부근으로 유사하게 나타났다. 또한 해당 input 구간을 masking했을 때 prediction이 크게 낮아졌다. 이는 모델이 해당 EEG 구간을 seizure 판단의 핵심 evidence로 사용하고 있음을 보여준다.

종합하면, 이번 MI 분석은 다음 메시지를 준다.

> SeizureTransformer는 특정 EEG 시간 구간의 seizure-like pattern을 input 단계에서 감지하고, 이를 Transformer latent representation 안의 seizure-relevant feature로 변환한 뒤, decoder를 통해 seizure probability를 출력한다.

---

## 8. 다음 단계 제안

현재 분석은 `abs(gradient * activation)` 기반이므로 중요도의 방향성을 구분하지 않는다. 후속 분석에서는 signed attribution을 추가하면 어떤 구간이 seizure score를 올리는지, 혹은 낮추는지 구분할 수 있다.

또한 masking 방식에 따라 결과가 달라질 수 있으므로, zero masking 외에도 channel mean, noise replacement, random masking baseline을 함께 비교하는 것이 좋다.

마지막으로 latent intervention에서는 suppress뿐 아니라 amplify, mean replacement, invert intervention을 비교하면 latent feature가 seizure evidence를 강화하는지 약화하는지 더 명확히 확인할 수 있다.

