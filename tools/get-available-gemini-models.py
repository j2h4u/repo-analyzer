#!/usr/bin/env python3

import os
import google.generativeai as genai
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import box

# --- Вспомогательная функция для форматирования чисел ---
def humanize_tokens(num):
    """Конвертирует число токенов в формат k/M."""
    if not num:
        return "0"
    
    if num >= 1_000_000:
        val = num / 1_000_000
        return f"{val:.1f}M".replace(".0M", "M")
    
    if num >= 1_000:
        val = num / 1_000
        return f"{val:.0f}k"
        
    return str(num)

# --- Основная логика ---
console = Console()
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found.")
    exit(1)

genai.configure(api_key=api_key)

try:
    models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            models.append(m)

    # Сортировка по ID
    models.sort(key=lambda x: x.name)

    # box.SIMPLE убирает лишние границы
    # header_style=None убирает цвета заголовков
    table = Table(box=box.SIMPLE, header_style=None)

    table.add_column("Display Name", no_wrap=True)
    table.add_column("ID", no_wrap=True)
    table.add_column("Ctx Win", no_wrap=True, justify="right") 
    table.add_column("Description")

    for m in models:
        description = m.description.strip() if m.description else ""
        limit = m.input_token_limit
        
        # Получаем текстовое представление (например, "128k" или "1M")
        limit_str = humanize_tokens(limit)
        
        # --- Логика раскрашивания ---
        if limit >= 1_000_000:
            # Ярко-циановый (bold делает цвет ярче/жирнее)
            limit_display = f"[bold cyan]{limit_str}[/bold cyan]"
        elif limit >= 500_000:
            # Зеленый
            limit_display = f"[green]{limit_str}[/green]"
        else:
            # Обычный цвет (меньше 500k)
            limit_display = limit_str
        
        table.add_row(
            m.display_name,
            m.name,
            limit_display, # Передаем уже раскрашенную строку
            description
        )

    console.print(table)

except Exception as e:
    print(f"Error: {e}")
