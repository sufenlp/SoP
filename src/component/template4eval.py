from src.component.template import get_eval_template_frame, num2English


class Template4EVAL:

    def __init__(self, used_character_list, temp_name="0_story_norefute_grammar", targetlm_name='LLaMA'):
        self.targetlm_name = targetlm_name

        self.eval_temp = get_eval_template_frame(temp_name)
        self.used_character_list = used_character_list
        self.character_num = len(self.used_character_list) + 1 if self.eval_temp.have_safe_character else len(self.used_character_list)

        # 获得 respond as A, B, C, and D
        self.used_character_name_list = [self.targetlm_name] if self.eval_temp.have_safe_character else []
        self.used_character_name_list += [used_character['character_name'] for used_character in self.used_character_list]
        self.used_character_name_list[-1] = f"and {self.used_character_name_list[-1]}" if len(self.used_character_name_list) > 1 else self.used_character_name_list[-1]
        self.used_character_str = ''
        for used_character_name in self.used_character_name_list:
            self.used_character_str += used_character_name + ', '
        if len(self.used_character_name_list) == 2:
            self.used_character_str = self.used_character_str.replace(", and", " and")
        self.used_character_str = self.used_character_str[:-2]

    def get_origin_template(self):
        if self.character_num == 1:
            prefix = self.eval_temp.single_prefix.replace('[SAFE_NAME]', self.targetlm_name)
            suffix = self.eval_temp.single_suffix.replace('[SAFE_NAME]', self.targetlm_name)
        elif self.character_num == 2:
            prefix = self.eval_temp.double_prefix.replace('[SAFE_NAME]', self.targetlm_name)
            suffix = self.eval_temp.double_suffix.replace('[SAFE_NAME]', self.targetlm_name)
        else:
            prefix = self.eval_temp.multi_prefix.replace('[SAFE_NAME]', self.targetlm_name)
            suffix = self.eval_temp.multi_suffix.replace('[SAFE_NAME]', self.targetlm_name)

        prefix = prefix.replace('[C_NUM]', num2English(self.character_num))
        suffix = suffix.replace('[USED_JB_NAME_LIST]', self.used_character_str)

        c_desp_list_str = ''
        c_do_list_str = f"{self.eval_temp.character_do.replace('[JB_NAME]', self.targetlm_name)}\n" if self.eval_temp.have_safe_character else ""
        tmp_safe_name = self.targetlm_name
        for idx, used_character in enumerate(self.used_character_list):
            used_character_name = used_character['character_name']
            used_character_desp = used_character['character_description']
            tmp_c_desp = self.eval_temp.character_desp.replace('[JB_NAME]', used_character_name)
            tmp_c_desp = tmp_c_desp.replace('[JB_DESCRIPTION]', used_character_desp)
            tmp_c_desp += self.eval_temp.refute.replace('[SAFE_NAME]', tmp_safe_name).replace('[JB_NAME]', used_character_name)

            c_desp_list_str += tmp_c_desp
            if self.eval_temp.refute_mode > 0:
                tmp_safe_name = used_character_name
            c_desp_list_str += '\n'

            tmp_c_do = self.eval_temp.character_do.replace('[JB_NAME]', used_character_name)
            c_do_list_str += tmp_c_do
            c_do_list_str += '\n'

        return prefix + '\n' + c_desp_list_str + suffix.replace('[CHARACTER DO]', c_do_list_str[:-1])  # 去掉最后的 \n


if __name__ == '__main__':
    template = Template4EVAL([{'character_name': "aaa", 'character_description': 'aaa is A'}, {'character_name': "bbb", 'character_description': 'bbb is B'}, {'character_name': "ccc", 'character_description': 'ccc is C'}])
    print(template.get_origin_template())
    print("-"*50)
    template = Template4EVAL([{'character_name': "aaa", 'character_description': 'aaa is A'}, {'character_name': "bbb", 'character_description': 'bbb is B'}])
    print(template.get_origin_template())
    print("-"*50)
    template = Template4EVAL([{'character_name': "aaa", 'character_description': 'aaa is A'}])
    print(template.get_origin_template())
    print("-"*50)

