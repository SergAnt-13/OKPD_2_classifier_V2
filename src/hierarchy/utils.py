def get_level(code: str) -> int:
    """
    Простейшая эвристика уровня ОКПД:
    2 цифры — класс
    4-5 — группа
    6+ — детализация
    """

    if len(code) <= 2:
        return 1
    elif len(code) <= 5:
        return 2
    else:
        return 3


def is_same_branch(code1: str, code2: str) -> bool:
    """
    Проверка принадлежности к одной ветке (упрощённо)
    """

    return code1[:2] == code2[:2]