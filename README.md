## Semestrální práce: Segmentace obrazu pomocí prahování
---
### Implementované metody
Projekt porovnává tři klasické metody prahování a jednu metodu založenou na neuronové síti:
1. **Uživatelský práh**: Manuální nastavení prahu uživatelem. Umožňuje kontrolu, ale je subjektivní a pracné.
2. **Otsu metoda**: Automatická statistická metoda, která minimalizuje vnitrotřídní rozptyl, účinná pro bimodální histogramy.
3. **Fuzzy přístup**: Využívá průměrnou intenzitu jako práh, je robustnější k šumu a neostrým hranicím.
4. **Neuronová síť U-Net**: Architektura hlubokého učení, která se učí segmentaci přímo z dat. Je robustní vůči šumu a variabilitě.
---
### Struktura projektu
| Soubor | Obsah a Účel | Metody |
| :--- | :--- | :--- |
| `Segmentace.ipynb` | **Notebook A** pro klasické metody. Interaktivní prostředí pro vizuální porovnání a IoU hodnocení (pro Režimy A, B). | Uživatelský práh, Otsu, Fuzzy |
| `U_Net_segmentace.ipynb` | **Notebook B** pro implementaci a trénink neuronové sítě U-Net pomocí PyTorch. | U-Net |
| `SP_Liskovsky_Prahovani.pdf` | Semestrální práce. Analýza, návrh řešení, implementace a evaluace výsledků. | Analýza a Srovnání Všech 4 Metod |
| `SP_Liskovsky_U_Net.pdf` | **Nově přidaná část (05.12.2025)** – detailní experimenty s U-Netem, interpretace výsledků a doporučení pro zlepšení. | U-Net (experimenty) |
| Experimenty.txt | **Nově přidaná část (05.12.2025)** – detailní experimenty - interpretace výsledků |
---
### Postup spuštění

Oba notebooky jsou navrženy pro spuštění v prostředí **Google Colab**.

#### 1. Notebook B (`U_Net_segmentace.ipynb`) - Trénink U-Net
1.  Otevřete notebook v Google Colab a připojte se ke **GPU runtime** (Runtime -> Change runtime type).
2.  Spusťte první buňku pro připojení Google Disku (`drive.mount`).
3.  **Důležité**: Upravte proměnnou `DATA_DIR` v buňce "0) Parametry..." tak, aby odkazovala na složku s datasetem párů obrázek/maska.
    ```python
    DATA_DIR = "/content/drive/MyDrive/Colab Notebooks/segment"
    ```
4.  Spusťte všechny následující buňky (Setup, UNet, Dataset, Trénink). Trénink probíhá po dobu 25 epoch.
5.  Poslední buňka "8) Vizualizace..." zobrazí výsledek U-Net predikce na validačním vzorku.

#### 2. Notebook A (`Segmentace.ipynb`) - Klasické metody
1.  Otevřete notebook v Google Colab.
2.  Spusťte první buňku (SETUP) s pomocnými funkcemi.
3.  Zvolte jeden z režimů (A, B nebo C) a spusťte jeho buňku:
    * **Režim A (Syntetika)**: Vygeneruje obraz s bimodálním histogramem pro kvantitativní testování.
    * **Režim B (Upload s GT)**: Umožní nahrát vlastní obrázek a odpovídající GT masku pro IoU srovnání.
    * **Režim C (Upload bez GT)**: Umožní nahrát libovolný obrázek pro čistě vizuální srovnání.
4.  Interaktivní posuvník "User T" vám umožní ladit uživatelský práh a pozorovat změny v IoU a vizuálním výsledku.

---
### Evaluace a Závěry
* **Syntetická data**: Klasické metody i U-Net dosahují téměř dokonalých výsledků (IoU $\approx 1$).
* **Reálná data**: Klasické metody selhávají. **U-Net** výrazně překonává ostatní (IoU $0.3-0.6$ vs. $0.05-0.1$ pro klasické metody), je robustnější a přesnější.
* **Experimenty (nově)**:  
  - Testováno vliv velikosti datasetu (200, 400, 800 vzorků), počtu epoch (25, 50, 100), batch size (4–32) a rozlišení (128 vs. 256 px).  
  - Nejlepší dosažené IoU ≈ **0.24** (N=800, 100 epoch, batch=16, IMG_SIZE=128).  
  - Výsledky ukazují, že výkon roste s větším datasetem a více epochami, ale klesá při příliš velkém rozlišení nebo batch size.  
  - Diskuze obsahuje doporučení: použití **Focal Loss**, silnějšího encoderu (ResNet34), augmentací (Albumentations) a scheduleru learning rate.  
* **Závěr**: Pro praktické nasazení v reálném světě je nezbytná neuronová síť **U-Net**.  
---
### 🆕 Aktualizace – 05.12.2025
- Přidána samostatná část projektu: `SP_Liskovsky_U_Net.pdf` s detailními experimenty.  
- Nově zahrnuty výsledky pro různé konfigurace (dataset size, epochy, batch, rozlišení).  
- Doplněna interpretace výsledků a doporučení pro budoucí práci.  
