"""
train_sore_v3.py
Script para treinamento do SOREModel v3.

CORRIGIDO: Este script agora carrega e usa corretamente o tokenizador BPE
treinado pelo 'tokenizer_pipeline.py' em vez do TokenizadorSimples.
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

    from src.core.tokenizer_pipeline import build_and_save_tokenizer

    from tokenizers import Tokenizer
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print("Verifique se os arquivos 'soreModel_v3.py' e 'tokenizer_pipeline.py' estão acessíveis no sys.path.")
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
    parser.add_argument('--tokenizer_dir', type=str, default='./tokenizer',
                        help='Diretório para salvar/carregar o tokenizador BPE')
    parser.add_argument('--vocab_size', type=int, default=52000,
                        help='Tamanho do vocabulário para o tokenizador BPE')
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
    return parser.parse_args()

class ConjuntoDeDadosTexto(Dataset):
    """
    Dataset que usa o tokenizador BPE da HuggingFace.
    Ele lida com a tokenização, truncamento e padding.
    """
    def __init__(self, textos, tokenizer, tamanho_maximo):
        self.textos = textos
        self.tokenizer = tokenizer
        self.tamanho_maximo = tamanho_maximo
        
        # Obter pad_id do tokenizer (definido em tokenizer_pipeline.py como "<|pad|>")
        self.pad_token_str = "<|pad|>"
        self.pad_id = self.tokenizer.token_to_id(self.pad_token_str)
        
        if self.pad_id is None:
            print(f"Aviso: Token de padding '{self.pad_token_str}' não encontrado. Usando token ID 0.")
            self.pad_id = 0
            self.pad_token_str = self.tokenizer.id_to_token(0) or "[PAD]"
            
        print(f"Dataset usando padding com ID: {self.pad_id} ('{self.pad_token_str}')")

    def __len__(self):
        return len(self.textos)
    
    def __getitem__(self, idx):
        texto = self.textos[idx]
        if not texto or not texto.strip(): # Lidar com textos vazios
            return torch.full((self.tamanho_maximo,), self.pad_id, dtype=torch.long)
            
        # Tokeniza
        tokens = self.tokenizer.encode(texto).ids
        
        # Trunca se for maior
        if len(tokens) > self.tamanho_maximo:
            tokens = tokens[:self.tamanho_maximo]
        
        # Preenche se for menor
        if len(tokens) < self.tamanho_maximo:
            tokens = tokens + [self.pad_id] * (self.tamanho_maximo - len(tokens))
            
        return torch.tensor(tokens, dtype=torch.long)

def obter_taxa_aprendizado(passo, passos_aquecimento, taxa_aprendizado_max, total_passos_decay=1000000):
    """Scheduler: Aquecimento linear e depois decaimento linear."""
    # Aquecimento linear
    if passo < passos_aquecimento:
        return taxa_aprendizado_max * (passo / passos_aquecimento)
    
    # Decaimento linear após o aquecimento
    # Garante que não decaia abaixo de uma fração da taxa máxima (ex: 10%)
    progresso_decay = (passo - passos_aquecimento) / max(1, total_passos_decay - passos_aquecimento)
    fator_decay = max(0.1, 1.0 - progresso_decay)
    
    return taxa_aprendizado_max * fator_decay

def treinar_epoca(modelo, carregador_dados, otimizador, dispositivo, epoca, args, passo_global, escalador=None):
    modelo.train()
    perda_total = 0.0
    taxa_atual = 0.0
    
    barra_progresso = tqdm(carregador_dados, desc=f'Época {epoca + 1}', leave=False)
    
    for indice_lote, lote in enumerate(barra_progresso):
        # Move o lote para o dispositivo
        entradas = lote.to(dispositivo)
        
        # Obtém as saídas do modelo
        # Usa precisão mista (AMP) se o escalador estiver disponível (CUDA)
        with torch.amp.autocast(device_type=dispositivo.type, enabled=(escalador is not None)):
            saidas = modelo(entradas) # (B, T, Vocab)
            
            # Prepara logits e rótulos para CrossEntropyLoss
            # Logits: (B, T-1, V) -> (B*(T-1), V)
            # Rótulos: (B, T) -> (B*(T-1),)
            # Nós prevemos o próximo token, então deslocamos
            logits_deslocados = saidas[..., :-1, :].contiguous()
            rotulos_deslocados = entradas[..., 1:].contiguous()
            
            # Calcula a perda
            funcao_perda = nn.CrossEntropyLoss()
            perda = funcao_perda(
                logits_deslocados.view(-1, logits_deslocados.size(-1)),
                rotulos_deslocados.view(-1)
            )
            
            # Ajusta a perda para acumulação de gradiente
            perda = perda / args.gradient_accumulation_steps
        
        # Passo de retropropagação
        if escalador is not None:
            escalador.scale(perda).backward()
        else:
            perda.backward()
        
        # Atualiza os pesos
        if (indice_lote + 1) % args.gradient_accumulation_steps == 0:
            # Atualiza a taxa de aprendizado antes do passo do otimizador
            taxa_atual = obter_taxa_aprendizado(passo_global, args.warmup_steps, args.learning_rate)
            for grupo in otimizador.param_groups:
                grupo['lr'] = taxa_atual

            # Recorte de gradiente
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
        
        # Atualiza a barra de progresso
        perda_item_acumulada = perda.item() * args.gradient_accumulation_steps
        perda_total += perda_item_acumulada
        perda_media = perda_total / (indice_lote + 1)
        barra_progresso.set_postfix({'perda': f'{perda_media:.4f}', 'lr': f'{taxa_atual:.2e}'})
        
        # Log no Weights & Biases
        if args.use_wandb and passo_global % 10 == 0 and (indice_lote + 1) % args.gradient_accumulation_steps == 0:
            import wandb
            wandb.log({
                'treino/perda_passo': perda_item_acumulada,
                'treino/perda_media_epoca': perda_media,
                'treino/taxa_aprendizado': taxa_atual,
                'epoca': epoca,
                'passo_global': passo_global
            })
        
        # Salva checkpoint
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
        
        # --- INÍCIO DA CORREÇÃO DO TOKENIZADOR ---
        
        caminho_tokenizer_dir = Path(args.tokenizer_dir)
        caminho_tokenizer_json = caminho_tokenizer_dir / 'tokenizer.json'
        
        # Verifica se o tokenizador BPE já existe
        if not caminho_tokenizer_json.exists():
            print(f"Tokenizador BPE não encontrado em {caminho_tokenizer_json}.")
            print("Preparando dados para treinar o tokenizador BPE...")
            
            # Salva os textos em um arquivo temporário para o treinamento do tokenizador
            caminho_textos_temp = caminho_tokenizer_dir / 'dados_treinamento_temp.txt'
            caminho_tokenizer_dir.mkdir(parents=True, exist_ok=True)
            
            with open(caminho_textos_temp, 'w', encoding='utf-8') as f:
                for texto in tqdm(textos_treinamento, desc="Salvando textos para tokenizer"):
                    f.write(texto + '\n')
            
            print('Treinando tokenizador BPE (isso pode demorar)...')
            build_and_save_tokenizer(
                [str(caminho_textos_temp)], 
                vocab_size=args.vocab_size, 
                save_dir=str(caminho_tokenizer_dir)
            )
            # Remove o arquivo temporário
            os.remove(caminho_textos_temp)
        else:
            print(f"Carregando tokenizador BPE existente de {caminho_tokenizer_json}...")
        
        # Carrega o tokenizador BPE (HuggingFace)
        tokenizador = Tokenizer.from_file(str(caminho_tokenizer_json))
        tamanho_vocabulario = tokenizador.get_vocab_size()
        print(f"Tokenizador BPE carregado. Vocabulário: {tamanho_vocabulario}")
        
        # --- FIM DA CORREÇÃO DO TOKENIZADOR ---
        
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
            vocab_size=tamanho_vocabulario, # CORRIGIDO: usa o vocab do BPE
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
                modelo, carregador_dados, otimizador, dispositivo, 
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