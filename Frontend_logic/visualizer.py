"""visualizer.py

**work split**

Justin: backbone of visualizer

Sara: frontend of visualizer

"""

import pygame
import numpy as np
import sys
import random
import heapq
import os
import time

# bring in the backend
try:
    import Backend_logic.grid_and_algorithm_search as backend
except Exception as e:
    print(f"Backend loaded with warning: {e}")

# map backend functions
initializing_fire_charcter = backend.initializing_fire_charcter
fire_state = backend.fire_state
fire_spreading = backend.fire_spreading
get_movement_cost = backend.get_movement_cost

# logic definitions

def precompute_fire_spread(initial_fire, steps=100, slow_spread_rate=0.03, fast_spread_rate=0.06):
    timeline = [initial_fire.copy()]
    current = initial_fire.copy()
    char_pos = (0,0,0) 
    for _ in range(1, steps):
        current = fire_spreading(current, char_pos, slow_spread_rate, fast_spread_rate)
        timeline.append(current.copy())
    return timeline

def heuristics(a, b, fire_grid):
    manhattan = abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])
    node_state = fire_grid[a[0], a[1], a[2]]
    if node_state == 2: return 1000
    elif node_state == 1: return manhattan + 100
    return manhattan

def a_star_3d(grid, stairwell_coords, main_exit_coord, fire_timeline, start_pos):
    L, W, H = grid.shape
    start = start_pos 
    
    directions = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]
    full_path = [start] 
    current_pos = start
    current_layer = H - 1
    previous_stair = None
    step_idx = 0
    total_cost = 0

    while current_layer >= 0:
        fire_grid = fire_timeline[min(step_idx, len(fire_timeline) - 1)]
        if current_layer == 0: goal = main_exit_coord
        else:
            candidates = [(x, y, z) for x, y, z in stairwell_coords
                         if z == current_layer and (not previous_stair or (x, y) != (previous_stair[0], previous_stair[1]))]
            if candidates:
                goal = min(candidates, key=lambda c: sum(get_movement_cost(fire_grid[nx, ny, nz])
                                  for nx in range(c[0]-1, c[0]+2) for ny in range(c[1]-1, c[1]+2) for nz in [c[2]]
                                  if 0 <= nx < L and 0 <= ny < W))
            else: goal = current_pos

        frontier = []
        heapq.heappush(frontier, (0, current_pos))
        came_from = {}
        g_score = np.full(grid.shape, np.inf)
        g_score[current_pos] = 0
        f_score = np.full(grid.shape, np.inf)
        f_score[current_pos] = heuristics(current_pos, goal, fire_grid)
        visited = np.zeros_like(grid, dtype=bool)
        exit_found = None

        while frontier:
            if not frontier: break
            _, current = heapq.heappop(frontier)
            x, y, z = current

            if visited[x, y, z]: continue
            visited[x, y, z] = True

            if current == goal:
                exit_found = current
                break
            
            fire_grid = fire_timeline[min(step_idx, len(fire_timeline) - 1)]
            for dx, dy in directions:
                nx, ny, nz = x + dx, y + dy, z
                if 0 <= nx < L and 0 <= ny < W and nz == current_layer:
                    move_cost = get_movement_cost(fire_grid[nx, ny, nz])
                    tentative_g = g_score[x, y, z] + move_cost
                    if tentative_g < g_score[nx, ny, nz]:
                        came_from[(nx, ny, nz)] = current
                        g_score[nx, ny, nz] = tentative_g
                        f_score[nx, ny, nz] = tentative_g + heuristics((nx, ny, nz), goal, fire_grid)
                        heapq.heappush(frontier, (f_score[nx, ny, nz], (nx, ny, nz)))
        
        if exit_found is None: break
        local_path = [goal]
        while local_path[-1] in came_from: local_path.append(came_from[local_path[-1]])
        local_path.reverse()

        for node in local_path[1:]:
            full_path.append(node)
            total_cost += get_movement_cost(fire_grid[node])
            step_idx += 1
        if current_layer == 0: break
        down_stair = (goal[0], goal[1], current_layer - 1)
        full_path.append(down_stair)
        total_cost += get_movement_cost(fire_grid[down_stair])
        step_idx += 1
        previous_stair = goal
        current_pos = down_stair
        current_layer -= 1
    
    return full_path, total_cost, [], start 

# animation helper
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

# visualizer

TILE_W = 64   
TILE_H = 32   
FLOOR_H = 50 

COLORS = {
    'background': (20, 20, 30),
    'tile_base': (50, 50, 60),
    'tile_border': (80, 80, 90),
    'safe': (150, 150, 150),
    'smoke': (255, 140, 0),
    'fire': (220, 20, 20),
    'stair': (60, 60, 200),
    'exit': (50, 200, 50),
    'agent': (0, 255, 255),
    'ui_bg': (10, 10, 15, 200), 
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
        
        self.main_exit_coord = (self.L - 1, self.W - 1, 0)
        
        self.SCREEN_WIDTH = 1000
        self.SCREEN_HEIGHT = 800
        self.OFFSET_X = self.SCREEN_WIDTH // 2 
        self.OFFSET_Y = 400 
        
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Dynamic Fire Pathfinding")
        self.font_sm = pygame.font.SysFont('Arial', 16)
        self.font_lg = pygame.font.SysFont('Arial', 24, bold=True)
        self.clock = pygame.time.Clock()
        self.grid = np.zeros((self.L, self.W, self.H), dtype=int)
        
        self.assets = {}
        self.animations = {}
        self.load_assets()
        
        self.step_delay = 250  
        self.last_step_time = 0
        self.state = 'SIMULATING' 
        
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
        slime_frames = self.load_sprite_sheet('assets/agent_slime.png', 32, 32, 3)
        if slime_frames: self.animations['agent'] = Animation(slime_frames, 150)
        
        # UI panel
        ui_path = 'assets/ui_panel.png'
        if os.path.exists(ui_path):
            self.assets['ui_panel'] = pygame.image.load(ui_path).convert_alpha()
        else:
            self.assets['ui_panel'] = None

        def load_tile(name, path, size=(TILE_W, TILE_H)):
            if os.path.exists(path):
                img = pygame.image.load(path).convert_alpha()
                self.assets[name] = pygame.transform.scale(img, size)
            else: self.assets[name] = None
                
        load_tile('floor', 'assets/floor.png')
        load_tile('stair', 'assets/stair.png', size=(TILE_W, FLOOR_H + TILE_H)) 
        load_tile('exit', 'assets/exit.png')
        load_tile('fire', 'assets/fire.png', size=(TILE_W, TILE_H + 20))
        load_tile('smoke', 'assets/smoke.png', size=(TILE_W, TILE_H + 10))

    def reset_simulation(self):
        self.current_seed = random.randint(0, 10000)
        random.seed(self.current_seed)
        np.random.seed(self.current_seed)
        
        self.fire_coord, self.char_start = initializing_fire_charcter(
            self.L, self.W, self.H, self.stairwell_coords, self.main_exit_coord)
        self.initial_fire_grid = fire_state(self.L, self.W, self.H, self.fire_coord)
        self.fire_timeline = precompute_fire_spread(self.initial_fire_grid, steps=100)
        
        random.seed(self.current_seed)
        np.random.seed(self.current_seed)
        self.path, self.total_path_cost, _, _ = a_star_3d(
            self.grid, self.stairwell_coords, self.main_exit_coord, 
            self.fire_timeline, self.char_start 
        )
        
        self.current_step = 0
        self.current_exposure = 0 
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
                if event.key == pygame.K_r:
                    self.reset_simulation()

    def update(self):
        if self.state == 'PAUSED': return 
        if 'agent' in self.animations: self.animations['agent'].update()

        now = pygame.time.get_ticks()
        if now - self.last_step_time > self.step_delay:
            self.last_step_time = now
            if self.path:
                self.current_step += 1
                if self.current_step >= len(self.path) or self.current_step >= len(self.fire_timeline):
                    self.playing = False
                    self.current_step = min(self.current_step, len(self.path) - 1)
                else:
                    fire_grid = self.fire_timeline[self.current_step]
                    pos = self.path[self.current_step]
                    node_val = fire_grid[pos[0], pos[1], pos[2]]
                    step_cost = get_movement_cost(node_val)
                    self.current_exposure += step_cost
                
                self.agent_pos = self.path[self.current_step]

    def draw_alpha_polygon(self, surface, color, points, alpha):
        lx, ly = min(p[0] for p in points), min(p[1] for p in points)
        rx, ry = max(p[0] for p in points), max(p[1] for p in points)
        w, h = rx - lx, ry - ly
        
        shape_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        local_points = [(p[0] - lx, p[1] - ly) for p in points]
        
        pygame.draw.polygon(shape_surf, (*color, alpha), local_points)
        pygame.draw.polygon(shape_surf, (*COLORS['tile_border'], alpha), local_points, 1)
        
        surface.blit(shape_surf, (lx, ly))

    def draw_iso_tile(self, surface, x, y, z, color, img=None, alpha=255):
        sx, sy = self.cart_to_iso(x, y, z)
        
        if img:
            temp_img = img.copy()
            if alpha < 255: temp_img.set_alpha(alpha)
            rect = temp_img.get_rect(center=(sx, sy + TILE_H//2))
            surface.blit(temp_img, rect)
            return sx, sy
        else:
            points = [
                (sx, sy), 
                (sx + TILE_W/2, sy + TILE_H/2),
                (sx, sy + TILE_H), 
                (sx - TILE_W/2, sy + TILE_H/2)
            ]
            
            if alpha < 255:
                self.draw_alpha_polygon(surface, color, points, alpha)
            else:
                pygame.draw.polygon(surface, color, points)
                pygame.draw.polygon(surface, COLORS['tile_border'], points, 1)
            
            return sx, sy

    def draw_simulation(self):
        fire_idx = min(self.current_step, len(self.fire_timeline) - 1)
        current_fire_grid = self.fire_timeline[fire_idx]
        agent_z = self.agent_pos[2]
        
        for z in range(self.H):
            if z > agent_z:
                alpha = 20  
                show_objects = False 
            else:
                alpha = 255 
                show_objects = True 
            
            for y in range(self.W):
                for x in range(self.L):
                    node_val = current_fire_grid[x, y, z]
                    
                    tile_col = COLORS['tile_base']
                    tile_img = self.assets.get('floor')
                    
                    is_special = False
                    if (x, y, z) == self.main_exit_coord: 
                        tile_col = COLORS['exit']
                        if self.assets.get('exit'): tile_img = self.assets['exit']
                        is_special = True
                    elif (x, y, z) in self.stairwell_coords: 
                        tile_col = COLORS['stair']
                        if self.assets.get('stair'): tile_img = self.assets.get('stair')
                        is_special = True
                    
                    cx, cy = self.draw_iso_tile(self.screen, x, y, z, tile_col, tile_img, alpha)
                    
                    if show_objects and not is_special:
                        obj_col = None
                        obj_img = None
                        if node_val == 2: 
                            obj_col = COLORS['fire']
                            if self.assets.get('fire'): obj_img = self.assets.get('fire')
                        elif node_val == 1: 
                            obj_col = COLORS['smoke']
                            if self.assets.get('smoke'): obj_img = self.assets.get('smoke')
                        
                        if obj_img:
                            temp = obj_img.copy()
                            rect = temp.get_rect(center=(cx, cy + TILE_H//2))
                            self.screen.blit(temp, rect)
                        elif obj_col:
                            self.draw_iso_tile(self.screen, x, y, z, obj_col, None, 255)

                    if (x, y, z) == self.agent_pos:
                        agent_img = self.assets.get('agent')
                        if 'agent' in self.animations:
                            frame = self.animations['agent'].get_frame()
                            rect = frame.get_rect(center=(cx, cy + TILE_H//2))
                            rect.bottom = cy + TILE_H - 5 
                            self.screen.blit(frame, rect)
                        elif agent_img:
                            rect = agent_img.get_rect(center=(cx, cy + TILE_H//2))
                            rect.bottom = cy + TILE_H - 5
                            self.screen.blit(agent_img, rect)
                        else:
                            pts = [
                                (cx, cy - 20), (cx + 10, cy - 10), 
                                (cx, cy), (cx - 10, cy - 10)
                            ]
                            pygame.draw.polygon(self.screen, COLORS['agent'], pts)
                            pygame.draw.polygon(self.screen, (255,255,255), pts, 2)

    def draw_legend(self):
        # KEYWORDS
        ui_rect = pygame.Rect(10, 10, 240, 220) # bigger to fit agent icon
        
        if self.assets.get('ui_panel'):
            panel_img = pygame.transform.scale(self.assets['ui_panel'], ui_rect.size)
            self.screen.blit(panel_img, ui_rect.topleft)
        else:
            panel = pygame.Surface(ui_rect.size, pygame.SRCALPHA)
            panel.fill(COLORS['ui_bg'])
            self.screen.blit(panel, ui_rect.topleft)
            
        x = ui_rect.left + 25
        y = ui_rect.top + 25
        line_height = 20
        
        self.screen.blit(self.font_lg.render("KEYWORDS", True, COLORS['text']), (x, y))
        y += 40
        
        def draw_entry(text, color, img_key=None, y_pos=0):
            # agent icon
            icon = None
            # agent animation
            if img_key == 'agent' and 'agent' in self.animations:
                icon = self.animations['agent'].frames[0]
            elif img_key and self.assets.get(img_key):
                icon = self.assets[img_key]
            
            if icon:
                icon = pygame.transform.scale(icon, (24, 24))
                self.screen.blit(icon, (x, y_pos))
            else:
                pygame.draw.rect(self.screen, color, (x, y_pos, 24, 24))
                pygame.draw.rect(self.screen, COLORS['tile_border'], (x, y_pos, 24, 24), 1)
            
            # text
            self.screen.blit(self.font_sm.render(text, True, COLORS['text']), (x + 35, y_pos + 4))

        # key
        draw_entry("Agent", COLORS['agent'], 'agent', y)
        y += line_height
        draw_entry("Safe: +1", COLORS['safe'], 'floor', y)
        y += line_height
        draw_entry("Smoke: +5", COLORS['smoke'], 'smoke', y)
        y += line_height
        draw_entry("Fire: +20", COLORS['fire'], 'fire', y)
        y += line_height + 10
        
        # stats explanation
        self.screen.blit(self.font_sm.render("Exposure = Sum of", True, COLORS['text']), (x, y))
        y += 20
        self.screen.blit(self.font_sm.render("all step costs", True, COLORS['text']), (x, y))

    def draw_ui(self):
        # right side UI
        ui_rect = pygame.Rect(self.SCREEN_WIDTH - 240, 10, 230, 200)
        
        if self.assets.get('ui_panel'):
            panel_img = pygame.transform.scale(self.assets['ui_panel'], ui_rect.size)
            self.screen.blit(panel_img, ui_rect.topleft)
        else:
            panel = pygame.Surface(ui_rect.size, pygame.SRCALPHA)
            panel.fill(COLORS['ui_bg'])
            self.screen.blit(panel, ui_rect.topleft)
        
        y = ui_rect.top + 25
        x_pad = 30

        def txt(t, off): 
            self.screen.blit(self.font_sm.render(t, True, COLORS['text']), (ui_rect.left + x_pad, y + off))

        txt(f"Step: {self.current_step}", 0)
        txt(f"Floor: {self.agent_pos[2]}", 25)
        txt(f"Exposure: {int(self.current_exposure)}", 50) 
        
        txt("[SPACE] Pause", 90)
        txt("[R] Reset", 115)

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
        self.draw_legend()
        
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