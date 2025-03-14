# import os
# import json
# # 导入Flask类库
# from flask import Flask
# from flask_cors import CORS
#
# # 创建应用实例
# app = Flask(__name__)
#
# def initial():
#     cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
#     anti_file = os.path.join(cur_dir, 'antisem.txt')
#
#     anti_dict = {}
#     sim_dict = {}
#
#     for line in open(anti_file, encoding='utf-8'):
#         line = line.strip().split(':')
#         wd = line[0]
#         antis = line[1].strip().split(';')
#         if wd not in anti_dict:
#             anti_dict[wd] = antis
#         else:
#             anti_dict[wd] += antis
#
#         for anti in antis:
#             if anti not in sim_dict:
#                 sim_dict[anti] = [i for i in antis if i != anti]
#             else:
#                 sim_dict[anti] += [i for i in antis if i != anti]
#     return anti_dict, sim_dict
#
# @app.route("/anti/overwrite")
# def over_write(dic):
#     cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
#     anti_file = os.path.join(cur_dir, 'test.txt')
#     f = open(anti_file, 'w', encoding='utf-8')
#     for k, v in dic.items():
#         s = ""
#         for i in v:
#             s += i + ";"
#         f.write(k + ":" + s[:-1] + "\n")
#
#     f.close()
#
# # 视图函数（路由）
# @app.route("/anti/dictionary")
# def read_txt():
#     anti_dict, sim_dict = None, None
#     if anti_dict is None or sim_dict is None:
#         anti_dict, sim_dict= initial()
#
#     anti_list = []
#     for k, v in anti_dict.items():
#         anti_str = ""
#         for j in v:
#             anti_str += j + ";"
#         m = {"word": k, "anti": anti_str[:-1]}
#         anti_list.append(m)
#
#     return json.dumps(anti_list, ensure_ascii=False)
#
# # 启动服务
# if __name__ == '__main__':
#     # r'/*' 是通配符，让本服务器所有的 URL 都允许跨域请求
#     CORS(app, resources=r'/*')
#     app.run(debug=True)
