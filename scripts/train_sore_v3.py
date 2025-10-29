"""
train_sore_v3.py
Script para treinamento do SOREModel v3.
"""
import os
import sys
import time
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

caminho_projeto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.basename(caminho_projeto) == 'src':
     caminho_projeto = os.path.dirname(caminho_projeto)
sys.path.insert(0, caminho_projeto)

try:
    from src.models.soreModel_v3 import SOREModel_v3, ModelConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print("Verifique se 'torch', 'datasets' e 'transformers' estão instalados.")
    print(f"Caminho do projeto atual: {caminho_projeto}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Treinar SOREModel v3')
    parser.add_argument('--dataset_name', type=str, default='wikitext',
                        help='Nome do dataset no HuggingFace')
    parser.add_argument('--dataset_config', type=str, default='wikitext-103-raw-v1',
                        help='Configuração do dataset')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_v3',
                        help='Diretório para salvar os checkpoints')
    parser.add_argument('--tokenizer_name', type=str, default='gpt2',
                        help='Nome do tokenizador pré-treinado do HuggingFace (ex: gpt2)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Tamanho do lote para treinamento')
    parser.add_argument('--context_size', type=int, default=512,
                        help='Tamanho do contexto (comprimento da sequência)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Número de épocas de treinamento')
    parser.add_argument('--learning_rate', type=float, default=6e-4,
                        help='Taxa de aprendizado')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Decaimento de peso')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Número de passos de aquecimento')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Número de passos de acumulação de gradiente')
    parser.add_argument('--save_steps', type=int, default=1000,
                        help='Salvar checkpoint a cada N passos')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Caminho para o checkpoint para continuar o treinamento')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Usar Weights & Biases para logging')
    parser.add_argument('--wandb_project', type=str, default='sore-model-v3',
                        help='Nome do projeto no Weights & Biases')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Número de workers para carregamento de dados (0 para desativar)')
    # Configs do Modelo
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--no_alibi', action='store_true', help='Desativa o ALiBi')
    parser.add_argument('--no_rmsnorm', action='store_true', help='Usa LayerNorm em vez de RMSNorm')
    # Argumentos para Destilação de Conhecimento (KD)
    parser.add_argument('--use_distillation', action='store_true',
                        help='Habilita o treinamento com destilação de conhecimento.')
    parser.add_argument('--teacher_model_name', type=str, default='gpt2-medium',
                        help='Nome do modelo professor no HuggingFace para destilação.')
    parser.add_argument('--distillation_alpha', type=float, default=0.5,
                        help='Peso da perda de destilação (0 a 1). Perda_total = alpha*perda_kd + (1-alpha)*perda_ce.')
    parser.add_argument('--distillation_temperature', type=float, default=2.0,
                        help='Temperatura para suavizar as distribuições de logits na destilação.')
    return parser.parse_args()

class ConjuntoDeDadosTexto(Dataset):
    """
    Dataset que tokeniza textos sob demanda usando um tokenizador do Hugging Face.
    """
    def __init__(self, textos, tokenizer, tamanho_maximo):
        self.textos = textos
        self.tokenizer = tokenizer
        self.tamanho_maximo = tamanho_maximo

    def __len__(self):
        return len(self.textos)
    
    def __getitem__(self, idx):
        texto = self.textos[idx]
        if not texto or not texto.strip():
            texto = self.tokenizer.eos_token  # Usa token de fim de texto para entradas vazias

        # Tokeniza o texto
        saida = self.tokenizer(
            texto,
            truncation=True,
            padding='max_length',
            max_length=self.tamanho_maximo,
            return_tensors='pt'  # Retorna tensores PyTorch
        )
        
        # Retorna os input_ids, removendo a dimensão do lote
        return saida['input_ids'].squeeze(0)

def obter_taxa_aprendizado(passo, passos_aquecimento, taxa_aprendizado_max, total_passos_decay=1000000):
    """Scheduler: Aquecimento linear e depois decaimento linear."""
    # Aquecimento linear
    if passo < passos_aquecimento:
        return taxa_aprendizado_max * (passo / passos_aquecimento)
    
    progresso_decay = (passo - passos_aquecimento) / max(1, total_passos_decay - passos_aquecimento)
    fator_decay = max(0.1, 1.0 - progresso_decay)
    
    return taxa_aprendizado_max * fator_decay

def treinar_epoca(modelo, modelo_professor, carregador_dados, otimizador, dispositivo, epoca, args, passo_global, escalador=None):
    modelo.train()
    if modelo_professor:
        modelo_professor.eval()

    perda_total = 0.0
    taxa_atual = 0.0
    barra_progresso = tqdm(carregador_dados, desc=f'Época {epoca + 1}', leave=False)
    
    for indice_lote, lote in enumerate(barra_progresso):
        entradas = lote.to(dispositivo)
        
        with torch.amp.autocast(device_type=dispositivo.type, enabled=(escalador is not None)):
            saidas = modelo(entradas) # (B, T, Vocab)
            logits_deslocados = saidas[..., :-1, :].contiguous()
            rotulos_deslocados = entradas[..., 1:].contiguous()
            
            # Perda padrão (Cross-Entropy)
            funcao_perda = nn.CrossEntropyLoss()
            perda_ce = funcao_perda(
                logits_deslocados.view(-1, logits_deslocados.size(-1)),
                rotulos_deslocados.view(-1)
            )

            perda = perda_ce
            perda_kd_item = 0.0

            # Se a destilação estiver habilitada, calcula a perda de destilação
            if args.use_distillation and modelo_professor:
                with torch.no_grad():
                    saidas_professor = modelo_professor(entradas).logits
                    logits_professor_deslocados = saidas_professor[..., :-1, :].contiguous()

                funcao_perda_kd = nn.KLDivLoss(reduction='batchmean')
                temp = args.distillation_temperature
                
                log_softmax_aluno = nn.functional.log_softmax(logits_deslocados / temp, dim=-1)
                softmax_professor = nn.functional.softmax(logits_professor_deslocados / temp, dim=-1)
                
                perda_kd = funcao_perda_kd(log_softmax_aluno, softmax_professor) * (temp ** 2)
                perda_kd_item = perda_kd.item()

                # Combina as perdas
                perda = args.distillation_alpha * perda_kd + (1 - args.distillation_alpha) * perda_ce

            perda_acumulada = perda / args.gradient_accumulation_steps
        
        if escalador is not None:
            escalador.scale(perda_acumulada).backward()
        else:
            perda_acumulada.backward()
        
        if (indice_lote + 1) % args.gradient_accumulation_steps == 0:
            taxa_atual = obter_taxa_aprendizado(passo_global, args.warmup_steps, args.learning_rate)
            for grupo in otimizador.param_groups:
                grupo['lr'] = taxa_atual
            if escalador is not None:
                escalador.unscale_(otimizador) # Desescala gradientes
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0) # Clip
                escalador.step(otimizador) # Passo do otimizador
                escalador.update() # Atualiza escala
            else:
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                otimizador.step()
            
            otimizador.zero_grad()
            passo_global += 1
        
        perda_item_acumulada = perda_acumulada.item() * args.gradient_accumulation_steps
        perda_total += perda_item_acumulada
        perda_media = perda_total / (indice_lote + 1)
        
        log_postfix = {'perda': f'{perda_media:.4f}', 'lr': f'{taxa_atual:.2e}'}
        if args.use_distillation:
            log_postfix['perda_kd'] = f'{perda_kd_item:.4f}'
        barra_progresso.set_postfix(log_postfix)
        
        if args.use_wandb and passo_global % 10 == 0 and (indice_lote + 1) % args.gradient_accumulation_steps == 0:
            import wandb
            wandb.log({
                'treino/perda_passo': perda_item_acumulada,
                'treino/perda_media_epoca': perda_media,
                'treino/perda_kd': perda_kd_item,
                'treino/taxa_aprendizado': taxa_atual,
                'epoca': epoca,
                'passo_global': passo_global
            })
        
        if passo_global > 0 and passo_global % args.save_steps == 0 and (indice_lote + 1) % args.gradient_accumulation_steps == 0:
            salvar_checkpoint(modelo, otimizador, passo_global, args, f'checkpoint_passo_{passo_global}')
    
    return perda_media, passo_global

def salvar_checkpoint(modelo, otimizador, passo, args, nome):
    diretorio_checkpoint = Path(args.output_dir) / nome
    diretorio_checkpoint.mkdir(parents=True, exist_ok=True)
    
    # Salva o modelo
    caminho_modelo = diretorio_checkpoint / 'modelo.pt'
    
    # Salva o estado do modelo, otimizador, passo e configuração
    estado_modelo = modelo.state_dict()
    estado_otimizador = otimizador.state_dict()
    
    torch.save({
        'modelo_state_dict': estado_modelo,
        'otimizador_state_dict': estado_otimizador,
        'passo': passo,
        'config': modelo.cfg.__dict__, # Salva a configuração do modelo
    }, caminho_modelo)
    
    # Salva a configuração (args) do treino
    caminho_config_treino = diretorio_checkpoint / 'config_treino.json'
    with open(caminho_config_treino, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    
    print(f'\nCheckpoint salvo em {diretorio_checkpoint}')

def main():
    args = parse_args()
    
    # Inicializa o Weights & Biases se habilitado
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=args)
    
    # Configura o dispositivo
    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando dispositivo: {dispositivo}')
    
    # Cria o diretório de saída
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Carrega o dataset
    print(f'Carregando dataset {args.dataset_name}/{args.dataset_config}...')
    try:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
        
        textos_treinamento = []
        for divisao in ['train', 'validation', 'test']:
            if divisao in dataset:
                # Filtra textos vazios ou nulos
                textos = [texto for texto in dataset[divisao]['text'] if texto and texto.strip()]
                textos_treinamento.extend(textos)
        
        if not textos_treinamento:
            raise ValueError("Nenhum texto válido encontrado no dataset")
            
        print(f'Carregado {len(textos_treinamento)} documentos de texto')
        
        # --- Carregando o tokenizador GPT-2 ---
        print(f"Carregando tokenizador '{args.tokenizer_name}' do Hugging Face...")
        tokenizador = AutoTokenizer.from_pretrained(args.tokenizer_name)
        
        # O GPT-2 não tem um token de padding por padrão, então usamos o token de fim de sentença
        if tokenizador.pad_token is None:
            tokenizador.pad_token = tokenizador.eos_token
            print(f"Token de padding não definido. Usando '{tokenizador.pad_token}' (ID: {tokenizador.pad_token_id})")

        tamanho_vocabulario = len(tokenizador)
        print(f"Tokenizador carregado. Vocabulário: {tamanho_vocabulario}")
        # --- Fim do carregamento do tokenizador ---
        
        # Cria o conjunto de dados
        conjunto_dados = ConjuntoDeDadosTexto(textos_treinamento, tokenizador, args.context_size)
        
        # Configura o carregador de dados
        carregador_dados = DataLoader(
            conjunto_dados,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        # Configuração do modelo
        configuracao = ModelConfig(
            vocab_size=tamanho_vocabulario, # Usa o tamanho do vocabulário do tokenizador carregado
            context_size=args.context_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=0.1,
            use_alibi=(not args.no_alibi),
            use_rmsnorm=(not args.no_rmsnorm)
        )
        
        print("Configuração do Modelo:")
        print(json.dumps(configuracao.__dict__, indent=2))
        
        # Inicializa o modelo
        modelo = SOREModel_v3(configuracao).to(dispositivo)
        
        # Inicializa o modelo professor, se a destilação estiver habilitada
        modelo_professor = None
        if args.use_distillation:
            print(f"Habilitada destilação de conhecimento com o professor: '{args.teacher_model_name}'")
            try:
                modelo_professor = AutoModelForCausalLM.from_pretrained(args.teacher_model_name).to(dispositivo)
                modelo_professor.eval() # Coloca em modo de avaliação
                print("Modelo professor carregado com sucesso.")
            except Exception as e:
                print(f"Erro ao carregar o modelo professor: {e}. A destilação será desativada.")
                args.use_distillation = False

        # Configura o otimizador
        otimizador = optim.AdamW(
            modelo.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Configura o escalador de gradiente para precisão mista (se CUDA disponível)
        escalador = torch.amp.GradScaler() if dispositivo.type == 'cuda' else None
        
        # Verifica se deve continuar o treinamento a partir de um checkpoint
        epoca_inicial = 0
        passo_global = 0
        if args.resume_from_checkpoint:
            caminho_checkpoint = Path(args.resume_from_checkpoint)
            if caminho_checkpoint.is_dir():
                caminho_checkpoint = caminho_checkpoint / 'modelo.pt'
                
            if caminho_checkpoint.exists():
                print(f'Resumindo treinamento a partir do checkpoint: {caminho_checkpoint}')
                checkpoint = torch.load(caminho_checkpoint, map_location=dispositivo)
                
                # Carrega o estado do modelo
                modelo.load_state_dict(checkpoint['modelo_state_dict'])
                
                # Carrega o estado do otimizador
                if 'otimizador_state_dict' in checkpoint:
                    otimizador.load_state_dict(checkpoint['otimizador_state_dict'])
                else:
                    print("Aviso: 'otimizador_state_dict' não encontrado no checkpoint.")

                passo_global = checkpoint.get('passo', 0)
                # Estimar a época inicial com base nos passos
                passos_por_epoca = len(carregador_dados) // args.gradient_accumulation_steps
                epoca_inicial = passo_global // passos_por_epoca
                
                print(f'Resumido treinamento. Passo global: {passo_global}, Época aprox.: {epoca_inicial}')
            else:
                print(f"Aviso: Checkpoint '{args.resume_from_checkpoint}' não encontrado. Começando do zero.")
        
        # Loop de treinamento
        print('Iniciando treinamento...')
        for epoca in range(epoca_inicial, args.epochs):
            tempo_inicio = time.time()
            
            # Treina por uma época
            perda_media, passo_global = treinar_epoca(
                modelo, modelo_professor, carregador_dados, otimizador, dispositivo, 
                epoca, args, passo_global, escalador
            )
            
            # Exibe métricas da época
            tempo_epoca = time.time() - tempo_inicio
            print(f'Época {epoca + 1}/{args.epochs} Concluída - Perda Média: {perda_media:.4f} - Tempo: {tempo_epoca:.2f}s')
            
            # Salva checkpoint no final de cada época
            salvar_checkpoint(modelo, otimizador, passo_global, args, f'checkpoint_epoca_{epoca + 1}')
        
        print('Treinamento concluído com sucesso!')
        
        # Salva o modelo final
        salvar_checkpoint(modelo, otimizador, passo_global, args, 'modelo_final')
        
    except Exception as e:
        print(f'Erro fatal durante o treinamento: {str(e)}')
        import traceback
        traceback.print_exc()
    finally:
        if args.use_wandb:
            wandb.finish()

if __name__ == '__main__':
    main()