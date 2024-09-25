import importlib


def get_eval_template_frame(temp_name="new.respond"):
    return importlib.import_module(f"src.component.template.{temp_name}")


def num2English(num):
    Eng_dict = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
            11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty"}
    return Eng_dict[num]


if __name__ == '__main__':
    print(get_eval_template_frame().single_prefix)
