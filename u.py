import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "text_classification_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict(text):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        print(logits)

    predicted_class = torch.argmax(logits, dim=-1).item()
    return predicted_class

text = "Оценка: 2 Комментарий: Много ошибок. В задании 1 правильный ответ «в», но он указан верно. В задании 3 правильный ответ только «А», а не «АБ». В задании 5 правильный ответ «Б», но выпуск облигаций (А) также может влиять на денежную массу. В задании 4 ответ верный, но в остальных есть неточности."
predicted_label = predict(text)
print(f"Текст: {text}")
print(f"Предсказанный класс: {predicted_label}")