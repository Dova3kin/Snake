import pygame
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import numpy as np
import os
from datetime import datetime

# ============================================================================
# ELEMENTS UI (BOUTONS & INPUTS)
# ============================================================================


class Button:
    """Bouton interactif simple pour Pygame."""

    def __init__(
        self,
        x,
        y,
        w,
        h,
        text,
        callback,
        color=(70, 130, 180),
        hover_color=(100, 160, 210),
    ):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback
        self.color = color
        self.hover_color = hover_color
        self.font = pygame.font.SysFont("arial", 20)
        self.is_hovered = False

    def draw(self, surface):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 2)

        text_surf = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return self.callback()
        return None


class InputBox:
    """Champ de saisie texte."""

    def __init__(self, x, y, w, h, text=""):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = pygame.Color("lightskyblue3")
        self.text = text
        self.txt_surface = pygame.font.SysFont("arial", 24).render(
            text, True, self.color
        )
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            self.color = (
                pygame.Color("dodgerblue2")
                if self.active
                else pygame.Color("lightskyblue3")
            )
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    return self.text
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                self.txt_surface = pygame.font.SysFont("arial", 24).render(
                    self.text, True, self.color
                )
        return None

    def draw(self, screen):
        screen.blit(self.txt_surface, (self.rect.x + 5, self.rect.y + 5))
        pygame.draw.rect(screen, self.color, self.rect, 2)


# ============================================================================
# TABLEAU DE BORD (DASHBOARD)
# ============================================================================


class Dashboard:
    """
    Interface graphique principale pour le monitoring de l'IA.
    Gère l'affichage du jeu, des graphiques, et la vision du CNN.
    """

    def __init__(self, width=1280, height=720):
        pygame.init()
        self.width = width
        self.height = height

        pygame.display.set_caption("Snake IA - Tableau de Bord")
        self.display = pygame.display.set_mode((width, height))

        # Layout
        self.sidebar_w = 250
        self.bottom_h = 40
        self.content_w = width - self.sidebar_w
        self.content_h = height - self.bottom_h

        # Quadrants pour l'affichage
        self.quad_w = self.content_w // 2
        self.quad_h = self.content_h // 2

        # États: RUNNING, PAUSED, MENU_SAVE, MENU_LOAD
        self.state = "RUNNING"
        self.paused = False

        # Surfaces
        self.game_surface = pygame.Surface((self.quad_w, self.quad_h))
        self.plot_surface = pygame.Surface((self.quad_w, self.quad_h))
        self.global_plot_surface = pygame.Surface((self.quad_w, self.quad_h))
        self.nn_surface = pygame.Surface((self.quad_w, self.quad_h))

        self.font = pygame.font.SysFont("arial", 15)
        self.title_font = pygame.font.SysFont("arial", 18, bold=True)

        self._init_ui_elements()
        self._init_matplotlib()

        self.info_text = ""

    def _init_ui_elements(self):
        """Initialisation des boutons et contrôles."""
        btn_x, btn_w, btn_h = 20, self.sidebar_w - 40, 35
        y_cursor = 50
        spacing = 15

        self.buttons = []
        self.lbl_controls = self.title_font.render("CONTRÔLES", True, (255, 255, 255))

        # Boutons Principaux
        self.buttons.append(
            Button(
                btn_x,
                y_cursor,
                btn_w,
                btn_h,
                "💾 Sauvegarder",
                lambda: "OPEN_SAVE",
                color=(46, 139, 87),
                hover_color=(60, 179, 113),
            )
        )
        y_cursor += btn_h + spacing

        self.buttons.append(
            Button(
                btn_x,
                y_cursor,
                btn_w,
                btn_h,
                "📂 Charger",
                lambda: "OPEN_LOAD",
                color=(70, 130, 180),
                hover_color=(100, 149, 237),
            )
        )
        y_cursor += btn_h + spacing

        self.buttons.append(
            Button(
                btn_x,
                y_cursor,
                btn_w,
                btn_h,
                "📷 Screenshot",
                lambda: "SCREENSHOT",
                color=(147, 112, 219),
                hover_color=(186, 85, 211),
            )
        )
        y_cursor += btn_h + spacing

        self.btn_export = Button(
            btn_x,
            y_cursor,
            btn_w,
            btn_h,
            "📊 Export Excel",
            lambda: "EXPORT",
            color=(34, 139, 34),
            hover_color=(50, 205, 50),
        )
        self.buttons.append(self.btn_export)
        y_cursor += btn_h + spacing * 2

        # Section Auto-Screen
        self.auto_screen_active = False
        self.screen_interval = 60
        self.lbl_autoscreen = self.font.render(
            "Auto-Screenshot:", True, (200, 200, 200)
        )
        y_cursor += 20

        self.btn_auto_screen = Button(
            btn_x,
            y_cursor,
            btn_w,
            btn_h,
            "Auto: OFF",
            lambda: "TOGGLE_AUTO_SCREEN",
            color=(100, 100, 100),
            hover_color=(120, 120, 120),
        )
        self.buttons.append(self.btn_auto_screen)
        y_cursor += btn_h + spacing

        self.lbl_interval = self.font.render("Interval (s):", True, (200, 200, 200))
        self.input_interval = InputBox(btn_x + 100, y_cursor - 5, 80, 30, text="60")
        y_cursor += btn_h + spacing * 2

        # Bouton Quitter
        self.buttons.append(
            Button(
                btn_x,
                self.height - 80,
                btn_w,
                btn_h,
                "❌ Quitter",
                lambda: "QUIT",
                color=(205, 92, 92),
                hover_color=(255, 99, 71),
            )
        )

        # Menu Sauvegarde
        self.btn_save_confirm = Button(
            0,
            0,
            120,
            35,
            "Valider",
            lambda: "CONFIRM_SAVE",
            color=(46, 139, 87),
            hover_color=(60, 179, 113),
        )
        self.btn_save_cancel = Button(
            0,
            0,
            120,
            35,
            "Annuler",
            lambda: "CANCEL_SAVE",
            color=(205, 92, 92),
            hover_color=(255, 99, 71),
        )
        self.input_box = InputBox(self.width // 2 - 100, self.height // 2, 200, 40)

        self.file_list = []
        self.scroll_y = 0

    def _init_matplotlib(self):
        """Configuration des graphiques avec réutilisation des axes pour éviter les fuites mémoire."""
        plt.style.use("dark_background")

        # Créer les figures et axes UNE SEULE fois
        self.fig_local, self.ax_local = plt.subplots(figsize=(5, 3.5), dpi=100)
        self.fig_global, self.ax_global = plt.subplots(figsize=(5, 3.5), dpi=100)

        self.surf_plot_local = None
        self.surf_plot_global = None
        self.snapshot = None

    def __del__(self):
        """Destructeur pour libérer les ressources Matplotlib."""
        try:
            plt.close(self.fig_local)
            plt.close(self.fig_global)
        except Exception:
            pass

    def update_info(self, n_games, total_time, epsilon, record):
        """Met à jour la barre d'info en bas."""
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        self.info_text = f"Parties: {n_games} | Record: {record} | Epsilon: {epsilon:.3f} | Temps: {hours:02d}:{minutes:02d}:{seconds:02d}"

    def update_game(self, game_surface):
        """Affiche le rendu du jeu (centré et mis à l'échelle)."""
        self.display.fill((20, 20, 25), (0, 0, self.quad_w, self.quad_h))

        gw, gh = game_surface.get_size()
        scale = min(self.quad_w / gw, self.quad_h / gh) * 0.95
        new_w, new_h = int(gw * scale), int(gh * scale)

        scaled_surf = pygame.transform.scale(game_surface, (new_w, new_h))
        pos_x = (self.quad_w - new_w) // 2
        pos_y = (self.quad_h - new_h) // 2

        self.display.blit(scaled_surf, (pos_x + self.sidebar_w, pos_y))
        pygame.draw.rect(
            self.display,
            (100, 100, 100),
            (pos_x + self.sidebar_w, pos_y, new_w, new_h),
            1,
        )
        self._draw_border(self.sidebar_w, 0, "🎮 Jeu en Direct (Agent 0)")

    def update_plots(self, scores, mean_scores, global_record):
        """Met à jour le graphique de session (réutilise l'axe existant)."""
        self.ax_local.clear()  # Clear l'axe au lieu de clf() sur la figure

        self.ax_local.plot(scores, label="Score", color="#00BFFF", linewidth=1.5)
        self.ax_local.plot(mean_scores, label="Moyenne", color="#FF6347", linewidth=2)
        self.ax_local.set_title(
            "Performance de la Session", fontsize=12, fontweight="bold"
        )
        self.ax_local.set_xlabel("Parties", fontsize=10)
        self.ax_local.set_ylabel("Score", fontsize=10)
        self.ax_local.legend(loc="upper left", fontsize=9)
        self.ax_local.grid(True, alpha=0.3)

        canvas = agg.FigureCanvasAgg(self.fig_local)
        canvas.draw()
        size = canvas.get_width_height()
        surf = pygame.image.fromstring(
            canvas.get_renderer().tostring_rgb(), size, "RGB"
        )
        self.surf_plot_local = pygame.transform.scale(surf, (self.quad_w, self.quad_h))

        self.display.blit(self.surf_plot_local, (self.sidebar_w + self.quad_w, 0))
        self._draw_border(
            self.sidebar_w + self.quad_w, 0, "📊 Graphique de Performance"
        )

        # Affichage placeholder global si vide
        if self.surf_plot_global is None:
            self.global_plot_surface.fill((30, 30, 30))
            wait_text = self.font.render(
                "En attente de données...", True, (100, 100, 100)
            )
            self.global_plot_surface.blit(
                wait_text, (self.quad_w // 2 - 60, self.quad_h // 2)
            )
            self.display.blit(self.global_plot_surface, (self.sidebar_w, self.quad_h))
            self._draw_border(self.sidebar_w, self.quad_h, "📈 Progression Globale")
        else:
            self.display.blit(self.surf_plot_global, (self.sidebar_w, self.quad_h))
            self._draw_border(self.sidebar_w, self.quad_h, "📈 Progression Globale")

    def update_global_plot(self, all_scores):
        """Met à jour le graphique global (nuage de points, réutilise l'axe existant)."""
        self.ax_global.clear()  # Clear l'axe au lieu de clf() sur la figure

        y = np.array(all_scores)
        x = np.arange(len(y))
        self.ax_global.scatter(x, y, s=8, alpha=0.6, c="#00CED1", edgecolors="none")
        self.ax_global.set_title(
            "Progression Globale (Nuage)", fontsize=12, fontweight="bold"
        )
        self.ax_global.set_ylabel("Score", fontsize=10)
        self.ax_global.set_xlabel("Parties Jouées", fontsize=10)
        self.ax_global.grid(True, alpha=0.3)

        canvas = agg.FigureCanvasAgg(self.fig_global)
        canvas.draw()
        size = canvas.get_width_height()
        surf = pygame.image.fromstring(
            canvas.get_renderer().tostring_rgb(), size, "RGB"
        )
        self.surf_plot_global = pygame.transform.scale(surf, (self.quad_w, self.quad_h))

        self.display.blit(self.surf_plot_global, (self.sidebar_w, self.quad_h))
        self._draw_border(self.sidebar_w, self.quad_h, "📈 Progression Globale")

    def update_nn(self, model, activations):
        """Visualise les activations du CNN (Entrées)."""
        self.nn_surface.fill((20, 20, 30))

        if len(activations) > 0 and activations[0] is not None:
            input_tensor = activations[0].cpu().numpy()
            if len(input_tensor.shape) == 4:
                input_tensor = input_tensor[0]

            ch_names = ["Corps (Vert)", "Tête (Bleu)", "Nourriture (Rouge)"]
            ch_colors = [(0, 200, 0), (50, 100, 255), (255, 50, 50)]

            # Dimensions dynamiques depuis le tenseur (C, H, W)
            tensor_h, tensor_w = input_tensor.shape[1], input_tensor.shape[2]

            # Calculer le scale pour que les 3 images rentrent dans le quadrant
            margin = 15
            available_w = self.quad_w - 40  # Marge de sécurité
            available_h = self.quad_h - 100  # Espace pour titre et légendes

            # Largeur totale = 3 * (tensor_w * scale) + 2 * margin
            max_scale_w = (available_w - 2 * margin) / (3 * tensor_w)
            max_scale_h = available_h / tensor_h
            scale = min(max_scale_w, max_scale_h, 6)  # Max 6 pour éviter pixelisation
            scale = max(1, int(scale))  # Au minimum 1

            img_w, img_h = tensor_w * scale, tensor_h * scale

            total_w = 3 * img_w + 2 * margin
            start_x = (self.quad_w - total_w) // 2

            for i in range(3):
                channel = input_tensor[i]
                surf = pygame.Surface((tensor_w, tensor_h))
                surf.fill((10, 10, 15))

                rows, cols = np.where(channel > 0)
                for r, c in zip(rows, cols):
                    surf.set_at((c, r), ch_colors[i])

                surf = pygame.transform.scale(surf, (img_w, img_h))
                x = start_x + i * (img_w + margin)
                y = 50
                pygame.draw.rect(
                    self.nn_surface,
                    ch_colors[i],
                    (x - 2, y - 2, img_w + 4, img_h + 4),
                    2,
                )
                self.nn_surface.blit(surf, (x, y))

                text = self.font.render(ch_names[i], True, ch_colors[i])
                text_rect = text.get_rect(center=(x + img_w // 2, y + img_h + 15))
                self.nn_surface.blit(text, text_rect)

        self.display.blit(self.nn_surface, (self.sidebar_w + self.quad_w, self.quad_h))
        self._draw_border(
            self.sidebar_w + self.quad_w, self.quad_h, "🧠 Vision du CNN (Entrées)"
        )

    def _take_screenshot(self):
        if not os.path.exists("screenshots"):
            os.makedirs("screenshots")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"screenshots/screenshot_{timestamp}.png"
        try:
            pygame.image.save(self.display, filename)
            print(f"📸 Screenshot sauvegardé: {filename}")
        except Exception as e:
            print(f"Erreur screenshot: {e}")

    def _draw_border(self, x, y, title):
        pygame.draw.rect(
            self.display, (60, 60, 70), (x, y, self.quad_w, self.quad_h), 2
        )
        title_bg = pygame.Surface((len(title) * 10 + 20, 25), pygame.SRCALPHA)
        title_bg.fill((30, 30, 40, 200))
        pos_y = y + self.quad_h - 30
        self.display.blit(title_bg, (x + 5, pos_y))
        text = self.title_font.render(title, True, (220, 220, 220))
        self.display.blit(text, (x + 10, pos_y + 3))

    def handle_input(self, event):
        """Gère les événements souris/clavier."""
        action = None

        if self.state == "RUNNING":
            for btn in self.buttons:
                res = btn.handle_event(event)
                if res == "OPEN_SAVE":
                    self.snapshot = self.display.copy()
                    self.state = "MENU_SAVE"
                    self.paused = True
                    self.input_box.text = "modele_sauvegarde"
                    self.input_box.active = True
                elif res == "OPEN_LOAD":
                    self.snapshot = self.display.copy()
                    self.state = "MENU_LOAD"
                    self.paused = True
                    if not os.path.exists("./model"):
                        os.makedirs("./model")
                    self.file_list = [
                        f for f in os.listdir("./model") if f.endswith(".pth")
                    ]
                elif res == "SCREENSHOT":
                    self._take_screenshot()
                elif res == "QUIT":
                    action = "QUIT"
                elif res == "EXPORT":
                    action = "EXPORT"
                elif res == "TOGGLE_AUTO_SCREEN":
                    self.auto_screen_active = not self.auto_screen_active
                    self.btn_auto_screen.text = (
                        f"Auto: {'ON' if self.auto_screen_active else 'OFF'}"
                    )
                    self.btn_auto_screen.color = (
                        (46, 139, 87) if self.auto_screen_active else (100, 100, 100)
                    )

            # Intervalle Input
            self.input_interval.handle_event(event)
            try:
                val = int(self.input_interval.text)
                if val > 0:
                    self.screen_interval = val
            except ValueError:
                pass

        elif self.state == "MENU_SAVE":
            if self.btn_save_confirm.handle_event(event) == "CONFIRM_SAVE":
                action = ("SAVE", self.input_box.text)
                self.state = "RUNNING"
                self.paused = False
            elif self.btn_save_cancel.handle_event(event) == "CANCEL_SAVE":
                self.state = "RUNNING"
                self.paused = False

            res = self.input_box.handle_event(event)
            if res:
                action = ("SAVE", res)
                self.state = "RUNNING"
                self.paused = False

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.state = "RUNNING"
                self.paused = False

        elif self.state == "MENU_LOAD":
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.state = "RUNNING"
                self.paused = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                dialog_h = min(80 + len(self.file_list) * 35 + 40, 400)
                dialog_y = (self.height - dialog_h) // 2
                start_y = dialog_y + 60

                for i, f in enumerate(self.file_list):
                    y = start_y + i * 35
                    if (
                        self.width // 2 - 190 < mx < self.width // 2 + 190
                        and y < my < y + 30
                    ):
                        action = ("LOAD", f)
                        self.state = "RUNNING"
                        self.paused = False
                        break
        return action

    def update(self):
        """Boucle de mise à jour graphique principale."""
        if self.state == "RUNNING":
            if self.surf_plot_local:
                self.display.blit(
                    self.surf_plot_local, (self.sidebar_w + self.quad_w, 0)
                )
            if self.surf_plot_global:
                self.display.blit(self.surf_plot_global, (self.sidebar_w, self.quad_h))

            # Sidebar
            pygame.draw.rect(
                self.display, (40, 40, 50), (0, 0, self.sidebar_w, self.height)
            )
            pygame.draw.line(
                self.display,
                (60, 60, 70),
                (self.sidebar_w, 0),
                (self.sidebar_w, self.height),
                2,
            )

            self.display.blit(self.lbl_controls, (20, 20))
            self.display.blit(
                self.lbl_autoscreen, (20, self.btn_auto_screen.rect.y - 25)
            )
            self.display.blit(self.lbl_interval, (20, self.input_interval.rect.y + 5))

            for btn in self.buttons:
                btn.draw(self.display)
            self.input_interval.draw(self.display)

            # Bottom Bar
            pygame.draw.rect(
                self.display,
                (30, 30, 35),
                (self.sidebar_w, self.content_h, self.content_w, self.bottom_h),
            )
            pygame.draw.line(
                self.display,
                (60, 60, 70),
                (self.sidebar_w, self.content_h),
                (self.width, self.content_h),
                2,
            )

            if self.info_text:
                txt_surf = self.title_font.render(self.info_text, True, (200, 200, 200))
                txt_rect = txt_surf.get_rect(
                    center=(
                        self.sidebar_w + self.content_w // 2,
                        self.content_h + self.bottom_h // 2,
                    )
                )
                self.display.blit(txt_surf, txt_rect)

        # Modales
        if self.state != "RUNNING" and self.snapshot:
            self.display.blit(self.snapshot, (0, 0))

        if self.state == "MENU_SAVE":
            self._draw_modal_bg()
            self._draw_save_dialog()

        elif self.state == "MENU_LOAD":
            self._draw_modal_bg()
            self._draw_load_dialog()

        pygame.display.flip()

    def _draw_modal_bg(self):
        s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        s.fill((0, 0, 0, 220))
        self.display.blit(s, (0, 0))

    def _draw_save_dialog(self):
        dialog_w, dialog_h = 450, 180
        dialog_x, dialog_y = (self.width - dialog_w) // 2, (self.height - dialog_h) // 2

        pygame.draw.rect(
            self.display,
            (50, 50, 60),
            (dialog_x, dialog_y, dialog_w, dialog_h),
            border_radius=10,
        )
        pygame.draw.rect(
            self.display,
            (100, 100, 120),
            (dialog_x, dialog_y, dialog_w, dialog_h),
            2,
            border_radius=10,
        )

        txt = self.title_font.render(
            "💾 Nom de la sauvegarde (ENTRÉE):", True, (255, 255, 255)
        )
        self.display.blit(txt, (dialog_x + 20, dialog_y + 25))

        self.input_box.rect.x = dialog_x + 50
        self.input_box.rect.y = dialog_y + 70
        self.input_box.rect.width = dialog_w - 100
        self.input_box.draw(self.display)

        hint = self.font.render("ÉCHAP pour annuler", True, (150, 150, 150))
        self.display.blit(
            hint, (dialog_x + (dialog_w - hint.get_width()) // 2, dialog_y + 120)
        )

        self.btn_save_confirm.rect.topleft = (dialog_x + 50, dialog_y + 115)
        self.btn_save_cancel.rect.topleft = (dialog_x + dialog_w - 170, dialog_y + 115)
        self.btn_save_confirm.draw(self.display)
        self.btn_save_cancel.draw(self.display)

    def _draw_load_dialog(self):
        dialog_w = 400
        dialog_h = min(80 + len(self.file_list) * 35 + 40, 400)
        dialog_x, dialog_y = (self.width - dialog_w) // 2, (self.height - dialog_h) // 2

        pygame.draw.rect(
            self.display,
            (50, 50, 60),
            (dialog_x, dialog_y, dialog_w, dialog_h),
            border_radius=10,
        )
        pygame.draw.rect(
            self.display,
            (100, 100, 120),
            (dialog_x, dialog_y, dialog_w, dialog_h),
            2,
            border_radius=10,
        )

        txt = self.title_font.render("📂 Choisir un modèle:", True, (255, 255, 255))
        self.display.blit(txt, (dialog_x + 20, dialog_y + 20))

        if not self.file_list:
            no_file = self.font.render("Aucune sauvegarde", True, (200, 100, 100))
            self.display.blit(
                no_file,
                (dialog_x + (dialog_w - no_file.get_width()) // 2, dialog_y + 70),
            )
        else:
            start_y = dialog_y + 60
            mx, my = pygame.mouse.get_pos()
            for i, f in enumerate(self.file_list):
                y = start_y + i * 35
                is_hover = (
                    dialog_x + 20 < mx < dialog_x + dialog_w - 20 and y < my < y + 30
                )
                if is_hover:
                    pygame.draw.rect(
                        self.display,
                        (70, 130, 180),
                        (dialog_x + 20, y, dialog_w - 40, 30),
                        border_radius=5,
                    )
                f_txt = self.font.render(
                    f"📄 {f}", True, (255, 255, 255) if is_hover else (200, 200, 200)
                )
                self.display.blit(f_txt, (dialog_x + 30, y + 6))

        hint = self.font.render("ÉCHAP pour annuler", True, (150, 150, 150))
        self.display.blit(
            hint,
            (dialog_x + (dialog_w - hint.get_width()) // 2, dialog_y + dialog_h - 30),
        )
