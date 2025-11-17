"""visualizer.py

**work split**

Justin: backbone of visualizer

Sara: 

"""

import pygame
import numpy as np
import sys
import random
import os
import time

# --- FIX PYTHON PATH SO WE CAN IMPORT Backend_logic ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- IMPORT BACKEND ---
from Backend_logic.grid_and_algorithm_search import (
    initializing_fire_charcter,
    fire_state,
    precompute_fire_spread,
    a_star_3d,
    get_movement_cost
)

# --- ANIMATION CLASS ---
class Animation:
    def __init__(self, frames, speed_ms):
        self.frames = frames
        self.speed = speed_ms
        self.last_update = 0
        self.frame_index = 0

    def update(self):
        now = pygame.time.get_ticks()
        if now - self.last_update > self.speed:
            self.last_update = now
            self.frame_index = (self.frame_index + 1) % len(self.frames)

    def get_frame(self):
        return self.frames[self.frame_index]

# --- VISUALIZER SETUP ---

TILE_W = 64   
TILE_H = 32   
FLOOR_H = 40 

COLORS = {
    'background': (25, 20, 35),
    'tile_base': (45, 40, 60),
    'tile_border': (80, 70, 100),
    'safe': (150, 150, 150),
    'smoke': (255, 165, 0),
    'fire': (255, 50, 50),
    'stair': (70, 70, 220),
    'exit': (50, 200, 50),
    'agent': (0, 255, 255),
    'ui_bg': (10, 10, 15, 220), 
    'ui_border': (100, 100, 120),
    'text': (240, 240, 240)
}

class FireSimulationApp:
    def __init__(self):
        pygame.init()
        
        self.L, self.W, self.H = 12, 9, 6
        self.stairwell_coords = [(0, 0, z) for z in range(self.H)] + \
                                [(0, self.W-1, z) for z in range(self.H)] + \
                                [(self.L-1, 0, z) for z in range(self.H)] + \
                                [(self.L-1, self.W-1, z) for z in range(self.H)]
        self.main_exit_coord = (self.L // 2, 0, 0)
        
        self.SCREEN_WIDTH = 1000
        self.SCREEN_HEIGHT = 800
        
        self.OFFSET_X = self.SCREEN_WIDTH // 2 
        self.OFFSET_Y = 350 
        
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Dynamic Fire Pathfinding")
        self.font_sm = pygame.font.SysFont('Arial', 16)
        self.font_lg = pygame.font.SysFont('Arial', 24, bold=True)
        self.clock = pygame.time.Clock()
        
        self.assets = {}
        self.animations = {}
        self.load_assets()
        
        self.step_delay = 1000  
        self.last_step_time = 0
        self.state = 'SIMULATING' 
        
        # placeholder grid for passing to backend
        self.grid = np.zeros((self.L, self.W, self.H), dtype=int)
        
        self.reset_simulation()

    def load_sprite_sheet(self, path, frame_width, frame_height, scale):
        if not os.path.exists(path): return None
        sheet = pygame.image.load(path).convert_alpha()
        frames = []
        for y in range(0, sheet.get_height(), frame_height):
            for x in range(0, sheet.get_width(), frame_width):
                frame = sheet.subsurface(pygame.Rect(x, y, frame_width, frame_height))
                frame = pygame.transform.scale(frame, (frame_width * scale, frame_height * scale))
                frames.append(frame)
        return frames

    def load_assets(self):
        slime_path = 'assets/agent_slime.png'
        if os.path.exists(slime_path):
            slime_frames = self.load_sprite_sheet(slime_path, 32, 32, 3)
            if slime_frames:
                self.animations['agent'] = Animation(slime_frames, 150)

        fire_path = 'assets/fire4.png'
        if os.path.exists(fire_path):
            fire_frames = self.load_sprite_sheet(fire_path, 64, 64, 2)
            if fire_frames:
                self.animations['fire'] = Animation(fire_frames, 100)

        def load_tile(name, path, size=(TILE_W, TILE_H)):
            if os.path.exists(path):
                img = pygame.image.load(path).convert_alpha()
                self.assets[name] = pygame.transform.scale(img, size)
            else:
                self.assets[name] = None

        tilesheet_path = 'assets/stone_tiles.png'
        if os.path.exists(tilesheet_path):
            sheet = pygame.image.load(tilesheet_path).convert_alpha()

            sheet_cols = 6
            sheet_rows = 5
            tile_w = sheet.get_width()  // sheet_cols
            tile_h = sheet.get_height() // sheet_rows

            col = 1
            row = 3

            rect = pygame.Rect(col * tile_w, row * tile_h, tile_w, tile_h)
            floor_img = sheet.subsurface(rect)

            floor_img = pygame.transform.smoothscale(floor_img, (TILE_W, TILE_H))
            self.assets['floor'] = floor_img
        else:
            self.assets['floor'] = None

        load_tile('stair', 'assets/stair.png', size=(TILE_W, FLOOR_H + TILE_H))
        load_tile('exit', 'assets/exit.png', size=(TILE_W, FLOOR_H + TILE_H))
        load_tile('smoke', 'assets/smoke.png', size=(TILE_W, FLOOR_H + TILE_H))
        load_tile('ui_panel', 'assets/ui_panel.png', size=(250, 250))





    def reset_simulation(self):
        self.current_seed = random.randint(0, 10000)
        random.seed(self.current_seed)
        np.random.seed(self.current_seed)
        
        self.fire_coord, self.char_start = initializing_fire_charcter(
            self.L, self.W, self.H, self.stairwell_coords, self.main_exit_coord)
        self.initial_fire_grid = fire_state(self.L, self.W, self.H, self.fire_coord)
        self.fire_timeline = precompute_fire_spread(self.initial_fire_grid, steps=100)
        
        # RESET SEED AGAIN
        # This ensures the backend A* generates the exact same start position internally
        random.seed(self.current_seed)
        np.random.seed(self.current_seed)
        
        self.path, self.cost, _, _ = a_star_3d(
            self.grid, 
            self.stairwell_coords, 
            self.main_exit_coord, 
            self.fire_timeline
        )
        
        self.current_step = 0
        self.last_step_time = pygame.time.get_ticks()
        self.state = 'SIMULATING'
        self.agent_pos = self.path[0] if self.path else self.char_start

    def cart_to_iso(self, x, y, z):
        iso_x = (x - y) * (TILE_W / 2)
        iso_y = (x + y) * (TILE_H / 2)
        screen_x = iso_x + self.OFFSET_X
        screen_y = iso_y + self.OFFSET_Y - (z * FLOOR_H) 
        return int(screen_x), int(screen_y)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: pygame.quit(); sys.exit()
                if event.key == pygame.K_SPACE:
                    self.state = 'PAUSED' if self.state == 'SIMULATING' else 'SIMULATING'
                if event.key == pygame.K_r: self.reset_simulation()

    def update(self):
        if self.state == 'PAUSED': return 
        if 'agent' in self.animations: self.animations['agent'].update()
        if 'fire' in self.animations: self.animations['fire'].update()

        now = pygame.time.get_ticks()
        if now - self.last_step_time > self.step_delay:
            self.last_step_time = now
            if self.path:
                self.current_step += 1
                if self.current_step >= len(self.path) or self.current_step >= len(self.fire_timeline):
                    self.playing = False
                    self.current_step = min(self.current_step, len(self.path) - 1)
                self.agent_pos = self.path[self.current_step]


    def draw_tile(self, surface, x, y, z, color_key, img_key=None, img_offset_y=0, alpha=255):
        sx, sy = self.cart_to_iso(x, y, z)
        points = [
            (sx, sy), (sx + TILE_W/2, sy + TILE_H/2),
            (sx, sy + TILE_H), (sx - TILE_W/2, sy + TILE_H/2)
        ]
        
        if self.assets['floor']:
            floor_img = self.assets['floor'].copy()
            if alpha < 255: floor_img.set_alpha(alpha)
            tile_rect = floor_img.get_rect(center=(sx, sy + TILE_H / 2))
            surface.blit(floor_img, tile_rect)
        else:
            if alpha < 255: pygame.draw.polygon(surface, (*COLORS[color_key], alpha), points, 1)
            else:
                pygame.draw.polygon(surface, COLORS[color_key], points)
                pygame.draw.polygon(surface, COLORS['tile_border'], points, 1)

        if img_key:
            img = None
            if img_key in self.animations: img = self.animations[img_key].get_frame()
            elif img_key in self.assets: img = self.assets[img_key]
            
            if img:
                if alpha < 255: 
                    img = img.copy()
                    img.set_alpha(alpha)
                img_rect = img.get_rect(center=(sx, sy - img_offset_y))
                surface.blit(img, img_rect)
        elif color_key not in ['tile_base', 'safe']:
             color = COLORS[color_key]
             if alpha < 255 and len(color) == 3: color = (*color, alpha)
             pygame.draw.circle(surface, color, (sx, int(sy-TILE_H/2)), 10)

    def draw_simulation(self):
        fire_idx = min(self.current_step, len(self.fire_timeline) - 1)
        current_fire_grid = self.fire_timeline[fire_idx]
        agent_z = self.agent_pos[2]
        
        for z in range(self.H):
            alpha = 10 if z > agent_z else 255
            for y in range(self.W):
                for x in range(self.L):
                    node_val = current_fire_grid[x, y, z]
                    img_key = None
                    color_key = 'tile_base' 
                    img_offset = 0
                    
                    if (x, y, z) == self.main_exit_coord: img_key, color_key = 'exit', 'exit'
                    elif (x, y, z) in self.stairwell_coords:
                        img_key, color_key = 'stair', 'stair'
                        img_offset = 0 #FLOOR_H / 2 
                    
                    if node_val == 2: img_key, color_key, img_offset = 'fire', 'fire', TILE_H 
                    elif node_val == 1: img_key, color_key, img_offset = 'smoke', 'smoke', TILE_H

                    self.draw_tile(self.screen, x, y, z, color_key, img_key, img_offset, alpha)

                    if (x, y, z) == self.agent_pos:
                        self.draw_tile(self.screen, x, y, z, 'agent', 'agent', TILE_H, alpha=255)

    def draw_ui(self):
        ui_rect = pygame.Rect(self.SCREEN_WIDTH - 240, 10, 230, 200)

        if self.assets.get('ui_panel'):
            panel_img = pygame.transform.scale(self.assets['ui_panel'], ui_rect.size)
            self.screen.blit(panel_img, ui_rect.topleft)
        else:
            panel = pygame.Surface(ui_rect.size, pygame.SRCALPHA)
            panel.fill(COLORS['ui_bg'])
            self.screen.blit(panel, ui_rect.topleft)

        y = ui_rect.top + 10
        def txt(t, off):
            self.screen.blit(
                self.font_sm.render(t, True, COLORS['text']),
                (ui_rect.left + 10, y + off)
            )

        txt(f"Step: {self.current_step}", 0)
        txt(f"Floor: {self.agent_pos[2]}", 20)
        txt(f"Path Cost: {int(self.cost)}", 40)
        txt("[SPACE] Pause  [R] Reset", 80)
        txt(f"Algorithm: A* 3D", 120)
        txt(f"Seed: {self.current_seed}", 140)


    def draw_pause_menu(self):
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        title = self.font_lg.render("PAUSED", True, COLORS['text'])
        self.screen.blit(title, title.get_rect(center=(self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2)))

    def draw(self):
        self.screen.fill(COLORS['background'])
        self.draw_simulation()
        self.draw_ui()
        if self.state == 'PAUSED': self.draw_pause_menu()
        pygame.display.flip()

    def run(self):
        while True:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)

if __name__ == "__main__":
    app = FireSimulationApp()
    app.run()