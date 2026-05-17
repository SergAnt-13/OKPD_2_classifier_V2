# test_lemmatization.py (временный)
from src.preprocessing.cleaner import TextCleaner

cleaner = TextCleaner()

examples = [
    "Конфеты в карамельной глазури",
    "Молоко сгущенное с сахаром 8.5%",
    "Трубы стальные электросварные прямошовные ГОСТ 10704-91",
    "Кефир 3.2% жирности ТУ 9222-001-12345678-2020",
    "Стул офисный ИЗО-9001",
    "Лист стальной оцинкованный 0.7х1250х2500 мм",
]

for text in examples:
    print(f"ИСХОДНЫЙ:     {text}")
    print(f"ЛЕММАТИЗИР.:  {cleaner.retrieval_view_normalized(text)}")
    print("-" * 70)
    
    
    