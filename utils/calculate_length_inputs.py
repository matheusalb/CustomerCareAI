import pandas as pd
import csv

if __name__ == '__main__':
    df_comentarios = pd.read_csv('./data/comentarios-criticos.csv', delimiter=',', on_bad_lines='skip')
    
    len_comentarios = df_comentarios['comentario'].str.len().mean()
    prompt_len= len(f"""
    Você é uma IA especializada em responder comentários negativos de um cliente a um restaurante.
    Sua tarefa é responder respeitosamente um comentário negativo de um cliente ao seu restaurante.
    Dado o comentário do cliente entre <>, gere um comentário de resposta de forma respeitosa, empática e não genérica, \
    convencendo o cliente que medidas serão tomadas \
    para resolver o seu problema e que ele poderá voltar a fazer pedidos no restaurante.
    Certifique-se de usar detalhes específicos do comentário do cliente.
    """)
    
    df_gen_1 = pd.read_csv('./data/comentarios-respostas-sample.csv', delimiter=',')
    len_respostas_1 = df_gen_1.resposta.str.len().mean()
    
    df_gen_2 = pd.read_csv('./data/comentarios-respostas-sample-prompt2.csv', delimiter=',')
    len_respostas_2 = df_gen_2.resposta.str.len().mean()
    
    
    print('Tamanho Médio Comentários:', len_comentarios)
    print('Tamanho Prompt:', prompt_len)
    print('Tamanho Médio Resposta prompt_1:', len_respostas_1)
    print('Tamanho Médio Resposta prompt_2:', len_respostas_2)
    print()
    print('Soma total:', len_comentarios+prompt_len+len_respostas_1)
    print('Tamanho médio total com prompt_2:', len_comentarios+prompt_len+len_respostas_2)
