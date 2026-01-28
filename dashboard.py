import pygame
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import numpy as np
import os
from datetime import datetime

# ============================================================================
# ÉLÉMENTS D'INTERFACE (BOUTONS & CHAMPS TEXTE)
# ============================================================================


class Bouton:
    """Un bouton simple sur lequel on peut cliquer."""

    def __init__(
        self,
        x,
        y,
        largeur,
        hauteur,
        texte,
        action,
        couleur=(70, 130, 180),
        couleur_survol=(100, 160, 210),
    ):
        self.rect = pygame.Rect(x, y, largeur, hauteur)
        self.texte = texte
        self.action = action
        self.couleur = couleur
        self.couleur_survol = couleur_survol
        self.font = pygame.font.SysFont("arial", 20)
        self.est_survole = False

    def dessiner(self, surface):
        c = self.couleur_survol if self.est_survole else self.couleur
        pygame.draw.rect(surface, c, self.rect)
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 2)

        surf_texte = self.font.render(self.texte, True, (255, 255, 255))
        rect_texte = surf_texte.get_rect(center=self.rect.center)
        surface.blit(surf_texte, rect_texte)

    def gerer_evenement(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.est_survole = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return self.action()
        return None


class BoiteSaisie:
    """Champ pour écrire du texte."""

    def __init__(self, x, y, largeur, hauteur, texte=""):
        self.rect = pygame.Rect(x, y, largeur, hauteur)
        self.couleur = pygame.Color("lightskyblue3")
        self.texte = texte
        self.surf_texte = pygame.font.SysFont("arial", 24).render(
            texte, True, self.couleur
        )
        self.actif = False

    def gerer_evenement(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.actif = not self.actif
            else:
                self.actif = False
            self.couleur = (
                pygame.Color("dodgerblue2")
                if self.actif
                else pygame.Color("lightskyblue3")
            )
        if event.type == pygame.KEYDOWN:
            if self.actif:
                if event.key == pygame.K_RETURN:
                    return self.texte
                elif event.key == pygame.K_BACKSPACE:
                    self.texte = self.texte[:-1]
                else:
                    self.texte += event.unicode
                self.surf_texte = pygame.font.SysFont("arial", 24).render(
                    self.texte, True, self.couleur
                )
        return None

    def dessiner(self, ecran):
        ecran.blit(self.surf_texte, (self.rect.x + 5, self.rect.y + 5))
        pygame.draw.rect(ecran, self.couleur, self.rect, 2)


# ============================================================================
# TABLEAU DE BORD PRINCIPAL
# ============================================================================


class Dashboard:
    """
    L'interface graphique qui affiche tout : le jeu, les graphiques, le cerveau.
    """

    def __init__(self, largeur=1280, hauteur=720):
        pygame.init()
        self.largeur = largeur
        self.hauteur = hauteur

        pygame.display.set_caption("Tableau de Bord - IA Snake")
        self.ecran = pygame.display.set_mode((largeur, hauteur))

        # Mise en page
        self.largeur_menu = 250
        self.hauteur_bas = 40
        self.largeur_contenu = largeur - self.largeur_menu
        self.hauteur_contenu = hauteur - self.hauteur_bas

        # On divise l'écran en 4 zones (quadrants)
        self.quad_l = self.largeur_contenu // 2
        self.quad_h = self.hauteur_contenu // 2

        # États possibles: RUNNING, PAUSED, MENU_SAVE, MENU_LOAD
        self.state = "RUNNING"
        self.en_pause = False

        # Surfaces de rendu (pour dessiner "hors écran" avant d'afficher)
        self.surface_jeu = pygame.Surface((self.quad_l, self.quad_h))
        self.surface_plot = pygame.Surface((self.quad_l, self.quad_h))
        self.surface_global = pygame.Surface((self.quad_l, self.quad_h))
        self.surface_nn = pygame.Surface((self.quad_l, self.quad_h))

        self.font = pygame.font.SysFont("arial", 15)
        self.font_titre = pygame.font.SysFont("arial", 18, bold=True)

        self._init_interface()
        self._init_graphiques()

        self.texte_info = ""

    def _init_interface(self):
        """On crée tous les boutons."""
        x, long, h = 20, self.largeur_menu - 40, 35
        y = 50
        espace = 15

        self.boutons = []
        self.lbl_controles = self.font_titre.render("CONTRÔLES", True, (255, 255, 255))

        # Boutons principaux
        self.boutons.append(
            Bouton(
                x,
                y,
                long,
                h,
                "Sauvegarder",
                lambda: "OUVRIR_SAVE",
                couleur=(46, 139, 87),
                couleur_survol=(60, 179, 113),
            )
        )
        y += h + espace

        self.boutons.append(
            Bouton(
                x,
                y,
                long,
                h,
                "Charger",
                lambda: "OUVRIR_LOAD",
                couleur=(70, 130, 180),
                couleur_survol=(100, 149, 237),
            )
        )
        y += h + espace

        self.boutons.append(
            Bouton(
                x,
                y,
                long,
                h,
                "Capture d'écran",
                lambda: "SCREENSHOT",
                couleur=(147, 112, 219),
                couleur_survol=(186, 85, 211),
            )
        )
        y += h + espace

        self.btn_export = Bouton(
            x,
            y,
            long,
            h,
            "Export Excel",
            lambda: "EXPORT",
            couleur=(34, 139, 34),
            couleur_survol=(50, 205, 50),
        )
        self.boutons.append(self.btn_export)
        y += h + espace * 2

        # Option Auto-Screenshot
        self.auto_screen_active = False
        self.screen_interval = 60
        self.lbl_autoscreen = self.font.render("Auto-Capture:", True, (200, 200, 200))
        y += 20

        self.btn_auto_screen = Bouton(
            x,
            y,
            long,
            h,
            "Auto: NON",
            lambda: "TOGGLE_AUTO_SCREEN",
            couleur=(100, 100, 100),
            couleur_survol=(120, 120, 120),
        )
        self.boutons.append(self.btn_auto_screen)
        y += h + espace

        self.lbl_interval = self.font.render("Intervalle (s):", True, (200, 200, 200))
        self.input_interval = BoiteSaisie(x + 100, y - 5, 80, 30, texte="60")
        y += h + espace * 2

        # Quitter
        self.boutons.append(
            Bouton(
                x,
                self.hauteur - 80,
                long,
                h,
                "Quitter",
                lambda: "QUIT",
                couleur=(205, 92, 92),
                couleur_survol=(255, 99, 71),
            )
        )

        # Boutons pour les menus (Save/Load)
        self.btn_confirmer = Bouton(
            0,
            0,
            120,
            35,
            "Valider",
            lambda: "CONFIRMER",
            couleur=(46, 139, 87),
            couleur_survol=(60, 179, 113),
        )
        self.btn_annuler = Bouton(
            0,
            0,
            120,
            35,
            "Annuler",
            lambda: "ANNULER",
            couleur=(205, 92, 92),
            couleur_survol=(255, 99, 71),
        )
        self.boite_nom_fichier = BoiteSaisie(
            self.largeur // 2 - 100, self.hauteur // 2, 200, 40
        )

        self.liste_fichiers = []
        self.snapshot = None  # Capture de l'écran pour faire un fond flou

    def _init_graphiques(self):
        """On prépare Matplotlib."""
        plt.style.use("dark_background")
        self.fig_local, self.ax_local = plt.subplots(figsize=(5, 3.5), dpi=100)
        self.fig_global, self.ax_global = plt.subplots(figsize=(5, 3.5), dpi=100)

        self.img_plot_local = None
        self.img_plot_global = None

    def __del__(self):
        try:
            plt.close(self.fig_local)
            plt.close(self.fig_global)
        finally:
            pass

    def update_info(self, nb_parties, temps_total, epsilon, record):
        heures = int(temps_total // 3600)
        minutes = int((temps_total % 3600) // 60)
        secondes = int(temps_total % 60)
        self.texte_info = f"Parties: {nb_parties} | Record: {record} | Epsilon: {epsilon:.3f} | Temps: {heures:02d}:{minutes:02d}:{secondes:02d}"

    def update_game(self, surface_jeu):
        """Affiche le jeu de l'agent 0."""
        self.ecran.fill((20, 20, 25), (0, 0, self.quad_l, self.quad_h))

        gl, gh = surface_jeu.get_size()
        echelle = min(self.quad_l / gl, self.quad_h / gh) * 0.95
        nl, nh = int(gl * echelle), int(gh * echelle)

        surf_echelle = pygame.transform.scale(surface_jeu, (nl, nh))
        pos_x = (self.quad_l - nl) // 2
        pos_y = (self.quad_h - nh) // 2

        self.ecran.blit(surf_echelle, (pos_x + self.largeur_menu, pos_y))
        pygame.draw.rect(
            self.ecran, (100, 100, 100), (pos_x + self.largeur_menu, pos_y, nl, nh), 1
        )
        self._dessiner_cadre(self.largeur_menu, 0, "🎮 Vue du Jeu")

    def update_plots(self, scores, moyennes, record):
        """Met à jour le graphique en temps réel."""
        self.ax_local.clear()
        self.ax_local.plot(scores, label="Score", color="#00BFFF", linewidth=1.5)
        self.ax_local.plot(moyennes, label="Moyenne", color="#FF6347", linewidth=2)
        self.ax_local.set_title(
            "Performance de la Session", fontsize=12, fontweight="bold"
        )
        self.ax_local.set_xlabel("Parties", fontsize=10)
        self.ax_local.set_ylabel("Score", fontsize=10)
        self.ax_local.legend(loc="upper left", fontsize=9)
        self.ax_local.grid(True, alpha=0.3)

        canvas = agg.FigureCanvasAgg(self.fig_local)
        canvas.draw()
        taille = canvas.get_width_height()
        raw_data = canvas.get_renderer().tostring_rgb()
        surf = pygame.image.fromstring(raw_data, taille, "RGB")
        self.img_plot_local = pygame.transform.scale(surf, (self.quad_l, self.quad_h))

        self.ecran.blit(self.img_plot_local, (self.largeur_menu + self.quad_l, 0))
        self._dessiner_cadre(self.largeur_menu + self.quad_l, 0, "Graphique")

        # Si pas encore de global, on affiche un texte
        if self.img_plot_global is None:
            self.surface_global.fill((30, 30, 30))
            txt = self.font.render("En attente de données...", True, (100, 100, 100))
            self.surface_global.blit(txt, (self.quad_l // 2 - 60, self.quad_h // 2))
            self.ecran.blit(self.surface_global, (self.largeur_menu, self.quad_h))
            self._dessiner_cadre(self.largeur_menu, self.quad_h, "Historique Global")
        else:
            self.ecran.blit(self.img_plot_global, (self.largeur_menu, self.quad_h))
            self._dessiner_cadre(self.largeur_menu, self.quad_h, "Historique Global")

    def update_global_plot(self, tous_les_scores):
        """Nuage de points global."""
        self.ax_global.clear()
        y = np.array(tous_les_scores)
        x = np.arange(len(y))
        self.ax_global.scatter(x, y, s=8, alpha=0.6, c="#00CED1", edgecolors="none")
        self.ax_global.set_title("Progression Totale", fontsize=12, fontweight="bold")
        self.ax_global.set_ylabel("Score", fontsize=10)
        self.ax_global.set_xlabel("Parties Jouées", fontsize=10)
        self.ax_global.grid(True, alpha=0.3)

        canvas = agg.FigureCanvasAgg(self.fig_global)
        canvas.draw()
        taille = canvas.get_width_height()
        raw = canvas.get_renderer().tostring_rgb()
        surf = pygame.image.fromstring(raw, taille, "RGB")
        self.img_plot_global = pygame.transform.scale(surf, (self.quad_l, self.quad_h))

        self.ecran.blit(self.img_plot_global, (self.largeur_menu, self.quad_h))
        self._dessiner_cadre(self.largeur_menu, self.quad_h, "Historique Global")

    def update_nn(self, activations):
        """Affiche ce que le robot 'voit'."""
        self.surface_nn.fill((20, 20, 30))

        if len(activations) > 0 and activations[0] is not None:
            entree = activations[0].cpu().numpy()
            if len(entree.shape) == 4:
                entree = entree[0]

            noms = ["Dégradé Corps", "Direction Tête", "Position Pomme"]
            couleurs = [(0, 200, 0), (50, 100, 255), (255, 50, 50)]

            th, tl = entree.shape[1], entree.shape[2]

            marge = 15
            dispo_l = self.quad_l - 40
            dispo_h = self.quad_h - 100

            max_scale_l = (dispo_l - 2 * marge) / (3 * tl)
            max_scale_h = dispo_h / th
            scale = int(min(max_scale_l, max_scale_h, 6))
            scale = max(1, scale)

            myl, myh = tl * scale, th * scale
            total_l = 3 * myl + 2 * marge
            start_x = (self.quad_l - total_l) // 2

            for i in range(3):
                canal = entree[i]
                surf = pygame.Surface((tl, th))
                surf.fill((10, 10, 15))

                rows, cols = np.where(canal > 0)
                for r, c in zip(rows, cols):
                    surf.set_at((c, r), couleurs[i])

                surf_zoom = pygame.transform.scale(surf, (myl, myh))
                x = start_x + i * (myl + marge)
                y = 50
                pygame.draw.rect(
                    self.surface_nn, couleurs[i], (x - 2, y - 2, myl + 4, myh + 4), 2
                )
                self.surface_nn.blit(surf_zoom, (x, y))

                txt = self.font.render(noms[i], True, couleurs[i])
                rt = txt.get_rect(center=(x + myl // 2, y + myh + 15))
                self.surface_nn.blit(txt, rt)

        self.ecran.blit(self.surface_nn, (self.largeur_menu + self.quad_l, self.quad_h))
        self._dessiner_cadre(self.largeur_menu + self.quad_l, self.quad_h, "Vision IA")

    def _take_screenshot(self):
        if not os.path.exists("screenshots"):
            os.makedirs("screenshots")
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        nom = f"screenshots/capture_{ts}.png"
        try:
            pygame.image.save(self.ecran, nom)
            print(f"📸 Screenshot : {nom}")
        except Exception as e:
            print(f"Erreur screenshot : {e}")

    def _dessiner_cadre(self, x, y, titre):
        pygame.draw.rect(self.ecran, (60, 60, 70), (x, y, self.quad_l, self.quad_h), 2)
        fond_titre = pygame.Surface((len(titre) * 10 + 20, 25), pygame.SRCALPHA)
        fond_titre.fill((30, 30, 40, 200))
        pos_y = y + self.quad_h - 30
        self.ecran.blit(fond_titre, (x + 5, pos_y))
        txt = self.font_titre.render(titre, True, (220, 220, 220))
        self.ecran.blit(txt, (x + 10, pos_y + 3))

    def handle_input(self, event):
        """Fonction appelée par agent.py pour gérer les clics."""
        action = None

        if self.state == "RUNNING":
            for btn in self.boutons:
                res = btn.gerer_evenement(event)
                if res == "OUVRIR_SAVE":
                    self.snapshot = self.ecran.copy()
                    self.state = "MENU_SAVE"
                    self.en_pause = True
                    self.boite_nom_fichier.texte = "mon_modele"
                    self.boite_nom_fichier.actif = True
                elif res == "OUVRIR_LOAD":
                    self.snapshot = self.ecran.copy()
                    self.state = "MENU_LOAD"
                    self.en_pause = True
                    if not os.path.exists("./model"):
                        os.makedirs("./model")
                    self.liste_fichiers = [
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
                    self.btn_auto_screen.texte = (
                        f"Auto: {'OUI' if self.auto_screen_active else 'NON'}"
                    )
                    self.btn_auto_screen.couleur = (
                        (46, 139, 87) if self.auto_screen_active else (100, 100, 100)
                    )

            self.input_interval.gerer_evenement(event)
            try:
                val = int(self.input_interval.texte)
                if val > 0:
                    self.screen_interval = val
            finally:
                pass

        elif self.state == "MENU_SAVE":
            if self.btn_confirmer.gerer_evenement(event) == "CONFIRMER":
                action = ("SAVE", self.boite_nom_fichier.texte)
                self.state = "RUNNING"
                self.en_pause = False
            elif self.btn_annuler.gerer_evenement(event) == "ANNULER":
                self.state = "RUNNING"
                self.en_pause = False

            res = self.boite_nom_fichier.gerer_evenement(event)
            if res:  # Entrée
                action = ("SAVE", res)
                self.state = "RUNNING"
                self.en_pause = False

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.state = "RUNNING"
                self.en_pause = False

        elif self.state == "MENU_LOAD":
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.state = "RUNNING"
                self.en_pause = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                h_dialog = min(80 + len(self.liste_fichiers) * 35 + 40, 400)
                y_dialog = (self.hauteur - h_dialog) // 2
                y_start = y_dialog + 60

                for i, f in enumerate(self.liste_fichiers):
                    y = y_start + i * 35
                    # Zone de clic approximative (centrée)
                    dw = 400
                    dx = (self.largeur - dw) // 2
                    if dx + 20 < mx < dx + dw - 20 and y < my < y + 30:
                        action = ("LOAD", f)
                        self.state = "RUNNING"
                        self.en_pause = False
                        break

        return action

    def update(self):
        """Mise à jour visuelle (boucle principale UI)."""
        if self.state == "RUNNING":
            if self.img_plot_local:
                self.ecran.blit(
                    self.img_plot_local, (self.largeur_menu + self.quad_l, 0)
                )
            if self.img_plot_global:
                self.ecran.blit(self.img_plot_global, (self.largeur_menu, self.quad_h))

            # Menu gauche
            pygame.draw.rect(
                self.ecran, (40, 40, 50), (0, 0, self.largeur_menu, self.hauteur)
            )
            pygame.draw.line(
                self.ecran,
                (60, 60, 70),
                (self.largeur_menu, 0),
                (self.largeur_menu, self.hauteur),
                2,
            )

            self.ecran.blit(self.lbl_controles, (20, 20))
            self.ecran.blit(self.lbl_autoscreen, (20, self.btn_auto_screen.rect.y - 25))
            self.ecran.blit(self.lbl_interval, (20, self.input_interval.rect.y + 5))

            for btn in self.boutons:
                btn.dessiner(self.ecran)
            self.input_interval.dessiner(self.ecran)

            # Barre du bas
            pygame.draw.rect(
                self.ecran,
                (30, 30, 35),
                (
                    self.largeur_menu,
                    self.hauteur_contenu,
                    self.largeur_contenu,
                    self.hauteur_bas,
                ),
            )
            pygame.draw.line(
                self.ecran,
                (60, 60, 70),
                (self.largeur_menu, self.hauteur_contenu),
                (self.largeur, self.hauteur_contenu),
                2,
            )

            if self.texte_info:
                surf = self.font_titre.render(self.texte_info, True, (200, 200, 200))
                rect = surf.get_rect(
                    center=(
                        self.largeur_menu + self.largeur_contenu // 2,
                        self.hauteur_contenu + self.hauteur_bas // 2,
                    )
                )
                self.ecran.blit(surf, rect)

        # Modales
        if self.state != "RUNNING" and self.snapshot:
            self.ecran.blit(self.snapshot, (0, 0))
            self._dessiner_fond_modal()

        if self.state == "MENU_SAVE":
            self._dessiner_dialogue_save()
        elif self.state == "MENU_LOAD":
            self._dessiner_dialogue_load()

        pygame.display.flip()

    def _dessiner_fond_modal(self):
        s = pygame.Surface((self.largeur, self.hauteur), pygame.SRCALPHA)
        s.fill((0, 0, 0, 220))
        self.ecran.blit(s, (0, 0))

    def _dessiner_dialogue_save(self):
        w, h = 450, 180
        x, y = (self.largeur - w) // 2, (self.hauteur - h) // 2

        pygame.draw.rect(self.ecran, (50, 50, 60), (x, y, w, h), border_radius=10)
        pygame.draw.rect(self.ecran, (100, 100, 120), (x, y, w, h), 2, border_radius=10)

        txt = self.font_titre.render("Nom de la sauvegarde :", True, (255, 255, 255))
        self.ecran.blit(txt, (x + 20, y + 25))

        self.boite_nom_fichier.rect.x = x + 50
        self.boite_nom_fichier.rect.y = y + 70
        self.boite_nom_fichier.rect.width = w - 100
        self.boite_nom_fichier.dessiner(self.ecran)

        indice = self.font.render("ECHAP pour annuler", True, (150, 150, 150))
        self.ecran.blit(indice, (x + (w - indice.get_width()) // 2, y + 120))

        self.btn_confirmer.rect.topleft = (x + 50, y + 115)
        self.btn_annuler.rect.topleft = (x + w - 170, y + 115)
        self.btn_confirmer.dessiner(self.ecran)
        self.btn_annuler.dessiner(self.ecran)

    def _dessiner_dialogue_load(self):
        w = 400
        h = min(80 + len(self.liste_fichiers) * 35 + 40, 400)
        x, y = (self.largeur - w) // 2, (self.hauteur - h) // 2

        pygame.draw.rect(self.ecran, (50, 50, 60), (x, y, w, h), border_radius=10)
        pygame.draw.rect(self.ecran, (100, 100, 120), (x, y, w, h), 2, border_radius=10)

        txt = self.font_titre.render("Choisir un modèle :", True, (255, 255, 255))
        self.ecran.blit(txt, (x + 20, y + 20))

        if not self.liste_fichiers:
            vide = self.font.render("Aucune sauvegarde trouvée", True, (200, 100, 100))
            self.ecran.blit(vide, (x + (w - vide.get_width()) // 2, y + 70))
        else:
            y_start = y + 60
            mx, my = pygame.mouse.get_pos()
            for i, f in enumerate(self.liste_fichiers):
                yf = y_start + i * 35
                hover = (x + 20 < mx < x + w - 20) and (yf < my < yf + 30)
                if hover:
                    pygame.draw.rect(
                        self.ecran,
                        (70, 130, 180),
                        (x + 20, yf, w - 40, 30),
                        border_radius=5,
                    )

                ftxt = self.font.render(
                    f, True, (255, 255, 255) if hover else (200, 200, 200)
                )
                self.ecran.blit(ftxt, (x + 30, yf + 6))

        indice = self.font.render("ECHAP pour annuler", True, (150, 150, 150))
        self.ecran.blit(indice, (x + (w - indice.get_width()) // 2, y + h - 30))
