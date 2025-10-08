Kaggle - Retail Sales Dataset

URL : https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset


Données générées (pour fournisseurs/comptabilité)

Mockaroo : https://www.mockaroo.com/

Structure Excel Professionnelle : Projet Finance Multi-Boutiques
📁 Architecture complète du projet
📦 PROJET_FINANCE_RETAIL/
│
├── 📊 00_MASTER_CONTROLLER.xlsx
│   └── Tableau de bord central avec liens vers tous les fichiers
│
├── 📂 01_DONNEES_SOURCES/
│   ├── Ventes_Raw.xlsx
│   ├── Encaissements_Raw.xlsx
│   ├── Fournisseurs_Raw.xlsx
│   └── Referentiels.xlsx
│
├── 📂 02_BASE_DONNEES/
│   └── BDD_Consolidee.xlsx (fichier pivot central)
│
├── 📂 03_TRAITEMENTS/
│   ├── ETL_PowerQuery.xlsx
│   └── Calculs_Intermediaires.xlsx
│
├── 📂 04_REPORTING/
│   ├── Dashboard_Direction.xlsx
│   ├── Reporting_KPI.xlsx
│   ├── Reconciliation_CA.xlsx
│   └── Suivi_Fournisseurs.xlsx
│
├── 📂 05_ANALYSES/
│   ├── Analyse_Ecarts_Budget.xlsx
│   └── Analyse_Rentabilite_Boutiques.xlsx
│
└── 📂 06_DOCUMENTATION/
    ├── Guide_Utilisation.pdf
    ├── Dictionnaire_Donnees.xlsx
    └── Process_MAJ.docx

📊 DÉTAIL DES FICHIERS EXCEL
00_MASTER_CONTROLLER.xlsx
Objectif : Hub central de navigation et actualisation
Structure :
┌─────────────────────────────────────────────┐
│  Onglet 1 : ACCUEIL                         │
├─────────────────────────────────────────────┤
│  • Logo entreprise                          │
│  • Date dernière mise à jour               │
│  • Boutons navigation vers chaque fichier  │
│  • Statut actualisation données            │
│  • Guide rapide utilisateur                │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Onglet 2 : LIENS_FICHIERS                  │
├─────────────────────────────────────────────┤
│  Tableau avec hyperliens vers :            │
│  - Tous les fichiers du projet            │
│  - Description de chaque fichier           │
│  - Responsable                             │
│  - Fréquence mise à jour                   │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Onglet 3 : PLANNING_MAJ                     │
├─────────────────────────────────────────────┤
│  • Calendrier des clôtures mensuelles      │
│  • Checklist tâches récurrentes            │
│  • Historique des mises à jour             │
└─────────────────────────────────────────────┘

📂 02_BASE_DONNEES/BDD_Consolidee.xlsx
Objectif : Base de données centrale structurée
Onglet 1 : VENTES
DateID_BoutiqueNom_BoutiqueN°_TransactionCatégorieMontant_HTTVAMontant_TTCMode_PaiementVendeur01/01/2025BTQ001Paris CentreVTE20250001Vêtements150.0030.00180.00CBJean D.01/01/2025BTQ002Lyon Part-DieuVTE20250002Accessoires45.009.0054.00EspècesMarie L.
Mise en forme :

Tableau structuré Excel (Ctrl+T)
Filtres automatiques
Mise en forme conditionnelle (montants > 500€ en vert)
Validation de données sur colonnes clés

Onglet 2 : ENCAISSEMENTS
Date_EncaissementID_BoutiqueRéférence_VenteMontant_EncaisséMode_PaiementStatut_CaisseDate_Remise_BanqueCommentaire01/01/2025BTQ001VTE20250001180.00CBValidé02/01/2025-01/01/2025BTQ002VTE2025000254.00EspècesValidé02/01/2025-
Onglet 3 : FOURNISSEURS
N°_FactureDate_FactureFournisseurMontant_HTTVAMontant_TTCDate_ÉchéanceAxe_AnalytiqueStatutDate_PaiementFA202500105/01/2025Textile Pro5000.001000.006000.0005/02/2025ACHATS_MARCHANDISESPayé03/02/2025FA202500210/01/2025EDF850.00170.001020.0010/02/2025CHARGES_ENERGIEImpayé-
Onglet 4 : BUDGET
MoisID_BoutiqueNom_BoutiqueCA_BudgetCharges_BudgetMarge_BudgetCommentairejanv.-25BTQ001Paris Centre800004500035000Soldes hiverjanv.-25BTQ002Lyon Part-Dieu650003800027000-
Onglet 5 : REF_BOUTIQUES
ID_BoutiqueNom_BoutiqueVilleRégionSurface_m2Date_OuvertureResponsableStatutBTQ001Paris CentreParisIDF12001/03/2020Sophie MartinActifBTQ002Lyon Part-DieuLyonAURA9515/06/2021Marc DupontActif
Onglet 6 : REF_AXES_ANALYTIQUES
Code_AxeLibelléCatégorieDépartementACH_MARCAchats MarchandisesCoût directCommercialCHG_PERSCharges PersonnelCoût structureRHCHG_ENERCharges ÉnergieCoût structureAdmin
Onglet 7 : CALENDRIER
DateAnnéeTrimestreMoisMois_NumSemaineJour_SemaineJour_OuvréPériode_Commerciale01/01/20252025T1Janvier11MercrediOuiSoldes Hiver
Formules utiles :
excelAnnée: =ANNEE([@Date])
Mois: =TEXTE([@Date];"mmmm")
Trimestre: ="T"&ARRONDI.SUP(MOIS([@Date])/3;0)
Jour_Ouvré: =SI(OU(JOURSEM([@Date];2)>5);Faux;Vrai)

📂 03_TRAITEMENTS/ETL_PowerQuery.xlsx
Objectif : Nettoyage et transformation des données
Onglet 1 : POWER_QUERY_CONNEXIONS
Connexions Power Query configurées :

Source_Ventes (lecture Ventes_Raw.xlsx)
Source_Encaissements
Source_Fournisseurs
Nettoyage automatique (suppression doublons, valeurs nulles)

Onglet 2 : TRANSFORMATIONS
Documentation des étapes Power Query :
ÉtapeDescriptionFormule M1Import donnéesExcel.Workbook(File.Contents("path"))2Promotion en-têtesTable.PromoteHeaders3Suppression doublonsTable.Distinct4Changement typeTable.TransformColumnTypes5Ajout colonne calculéeTable.AddColumn("Marge", each [CA]-[Achats])
Onglet 3 : REGLES_GESTION
RègleChampValidationAction si erreurR01Montant_TTC> 0Signaler en rougeR02Date_Encaissement>= Date_VenteAlerterR03Statut_Caisse"Validé" ou "En attente"Bloquer import

📂 04_REPORTING/Dashboard_Direction.xlsx
Objectif : Vue stratégique pour la direction
Onglet 1 : DASHBOARD_MENSUEL
┌────────────────────────────────────────────────────────┐
│  📊 REPORTING FINANCIER - JANVIER 2025                                 │
├────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │   CA     │ │  MARGE   │ │ PANIER   │ │TRANS.    ││
│  │ 245K€    │ │  98K€    │ │  85€     │ │2,890     ││
│  │ +12% ▲   │ │  40%     │ │  +2% ▲   │ │+8% ▲     ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘│
│                                                        │
│  📈 ÉVOLUTION CA (Graphique courbe)                   │
│  [Ligne CA Réel vs Budget vs N-1]                    │
│                                                        │
│  🏪 TOP 5 BOUTIQUES (Histogramme horizontal)          │
│  Paris Centre    █████████████████ 85K€              │
│  Lyon Part-Dieu  ███████████ 58K€                    │
│  ...                                                   │
│                                                        │
│  🥧 RÉPARTITION CA PAR CATÉGORIE (Camembert)         │
│  Vêtements: 45% | Accessoires: 30% | Chaussures: 25%│
│                                                        │
└────────────────────────────────────────────────────────┘

📊 SLICERS (Segments) :
- Période : ☑️ Janvier | ☐ Février | ☐ Mars
- Boutique : ☑️ Toutes | ☐ Paris | ☐ Lyon
- Catégorie : ☑️ Toutes
Formules clés :
excel=SOMMEPROD((Ventes[Date]>=DebutMois)*(Ventes[Date]<=FinMois)*Ventes[Montant_TTC])

=SIERREUR(Réel/Budget-1;0)

=SI(Réel>=Budget;"✓ Objectif atteint";"⚠ Sous objectif")
Onglet 2 : SYNTHESE_BOUTIQUES
BoutiqueCA RéelCA BudgetÉcart% Réal.Marge €Marge %TransactionsPanier MoyenTendanceParis Centre85,000€80,000€+5,000€106%34,000€40%1,00085€▲Lyon Part-Dieu58,000€65,000€-7,000€89%22,000€38%72081€▼
Mise en forme conditionnelle :

Échelle de couleurs sur % Réalisation (rouge < 90% < jaune < 100% < vert)
Icônes tendance (flèches)
Barres de données sur CA Réel


📂 04_REPORTING/Reconciliation_CA.xlsx
Objectif : Rapprochement CA/Encaissements
Onglet 1 : RAPPROCHEMENT_QUOTIDIEN
DateBoutiqueCA DéclaréEncaissementsÉcart% ConcordanceStatut CaisseCommentaire15/01/25BTQ0013,250€3,250€0€100%Validé✓ OK15/01/25BTQ0022,180€2,150€-30€99%À vérifier⚠ Écart espèces
Formules :
excelÉcart: =[@[CA Déclaré]]-[@Encaissements]

% Concordance: =[@Encaissements]/[@[CA Déclaré]]

Statut: =SI([@Écart]=0;"✓ OK";SI(ABS([@Écart])<50;"⚠ Écart mineur";"❌ Écart significatif"))
Onglet 2 : ALERTES
Filtre automatique sur :

Écarts > 50€
Statut caisse "Non validé"
Ancienneté > 3 jours

Tableau de suivi des actions correctives

📂 04_REPORTING/Suivi_Fournisseurs.xlsx
Objectif : Gestion comptabilité fournisseurs
Onglet 1 : FACTURES_ECHUES
N° FactureFournisseurMontant TTCDate ÉchéanceJours RetardAxe AnalytiqueActionFA2025045Textile Pro8,500€20/01/2510ACHATS_MARC🔴 Payer urgentFA2025052EDF1,200€25/01/255CHG_ENERGIE🟡 À planifier
Calcul automatique :
excelJours_Retard: =SI(AUJOURDHUI()>[@[Date Échéance]];AUJOURDHUI()-[@[Date Échéance]];0)

Action: =SI([@[Jours Retard]]>7;"🔴 Payer urgent";SI([@[Jours Retard]]>0;"🟡 À planifier";"🟢 Dans les temps"))
Onglet 2 : ANALYSE_AXES
Tableau croisé dynamique :

Lignes : Axe Analytique
Valeurs : Somme Montants TTC
Colonnes : Mois
Graphique : Histogramme empilé

Onglet 3 : FACTURATION_B2B
N° Facture ClientClientMontant TTCDate FactureDate ÉchéanceEncaisséReste à EncaisserStatut LettrageFC2025001Entreprise A15,000€05/01/2505/02/2515,000€0€Lettré ✓FC2025002Entreprise B8,500€12/01/2512/02/250€8,500€Non lettré

📂 05_ANALYSES/Analyse_Ecarts_Budget.xlsx
Objectif : Analyse approfondie des performances
Onglet 1 : WATERFALL_CHART
Analyse des écarts par nature :

CA : +12,000€
Coût marchandises : -5,000€
Charges personnel : -2,000€
Autres charges : -1,500€
= Marge finale : +3,500€

(Graphique en cascade Excel)
Onglet 2 : VARIANCE_ANALYSIS
KPIBudgetRéelÉcart €Écart %Favorable/DéfavorableCA Total245,000€257,000€+12,000€+4.9%✅ FavorableMarge Brute98,000€101,500€+3,500€+3.6%✅ FavorableCharges Fixes45,000€48,000€-3,000€-6.7%❌ Défavorable

🎨 Charte graphique recommandée
Palette de couleurs :
- Bleu principal : #2E5090 (titres, bordures)
- Bleu clair : #5B9BD5 (graphiques)
- Vert : #70AD47 (indicateurs positifs)
- Rouge : #FF6B6B (alertes, indicateurs négatifs)
- Orange : #FFA500 (avertissements)
- Gris : #D9D9D9 (fond tableaux)
Typographie :

Titres : Calibri 14pt, Gras
Corps : Calibri 11pt
Tableaux : Calibri 10pt

Standards de mise en forme :

Bordures : Fines, gris foncé
En-têtes tableaux : Fond bleu, texte blanc, gras
Lignes alternées : Gris clair/blanc
Montants : Format comptabilité (€), 2 décimales


🔧 Formules Excel avancées essentielles
KPIs Dashboard :
excel// CA Mois en cours avec critères multiples
=SOMMEPROD((Ventes[Date]>=DATE(2025;1;1))*(Ventes[Date]<=DATE(2025;1;31))*(Ventes[Montant_TTC]))

// CA même période année précédente
=SOMMEPROD((Ventes[Date]>=DATE(2024;1;1))*(Ventes[Date]<=DATE(2024;1;31))*(Ventes[Montant_TTC]))

// Évolution vs N-1
=SIERREUR((CA_N - CA_N1)/CA_N1;0)

// Taux de réalisation budget
=SI(Budget<>0;CA_Reel/Budget;0)

// Panier moyen
=SOMMEPROD(Ventes[Montant_TTC])/NB.SI.ENS(Ventes[ID_Transaction];"<>")

// Rang boutique
=RANG([@CA];Tableau[CA];0)
Réconciliation :
excel// Vérification concordance
=SI(ABS(CA-Encaissement)<0.01;"OK";"ÉCART")

// Recherche transaction
=INDEX(Ventes[Montant];EQUIV([@Ref];Ventes[N°_Transaction];0))

// Statut lettrage
=SI(NB.SI(Encaissements[Ref_Facture];[@N°_Facture])>0;"Lettré";"Non lettré")
Analyse fournisseurs :
excel// Délai moyen paiement
=MOYENNE.SI(Fournisseurs[Statut];"Payé";Fournisseurs[Jours_Paiement])

// Montant échu non payé
=SOMME.SI.ENS(Fournisseurs[Montant_TTC];Fournisseurs[Date_Échéance];"<"&AUJOURDHUI();Fournisseurs[Statut];"Impayé")

// Top 5 fournisseurs
=GRANDE.VALEUR(Fournisseurs[Montant_Total];LIGNE())

📋 Checklist mise en production
Avant livraison :

 Tous les tableaux structurés (Ctrl+T)
 Validation données sur champs critiques
 Protection des onglets formules (sauf saisie)
 Liens relatifs (pas de chemins absolus)
 Test actualisation Power Query
 Vérification formules #N/A
 Cohérence format dates (JJ/MM/AAAA)
 Nom de plages définis pour formules complexes
 Commentaires sur cellules importantes
 Masquer colonnes calculs intermédiaires

Documentation :

 Mode d'emploi (PDF)
 Dictionnaire de données
 Procédure MAJ mensuelle
 Contact support


💡 Astuces professionnelles
Navigation rapide :

Créer des hyperliens entre onglets
Nommer les plages importantes (Ctrl+F3)
Table des matières sur premier onglet

Performance :

Limiter formules volatiles (AUJOURDHUI, ALEA)
Utiliser tableaux structurés vs plages
Power Query vs formules multiples
Calcul manuel si fichier lourd

Sécurité :

Sauvegarde quotidienne automatique
Versionning (v1.0, v1.1...)
Protection cellules avec mot de passe
Droits lecture seule pour utilisateurs finaux


Cette structure est prête à l'emploi et démontre une maîtrise professionnelle d'Excel pour des missions finance ! 🚀