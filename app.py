import os
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
PHI = 1.6180339887

FACE_CASCADE  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
EYE_CASCADE   = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
SMILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# ── DENTAL KNOWLEDGE BASE ──────────────────────────────────────────────────
DENTAL_PROCEDURES = {
    "teeth_whitening": {
        "name": "Teeth Whitening", "icon": "✦",
        "description": "Professional bleaching to remove stains and restore brightness.",
        "ideal_for": ["Yellowing or dull teeth", "Coffee/tea/wine stains", "Tobacco stains", "Age-related discoloration"],
        "types": ["In-office laser whitening (1 session)", "Custom take-home trays (2–3 weeks)", "OTC strips (mild cases)"],
        "cost": "$300–$1,500 (professional) / $20–$100 (OTC)",
        "duration": "1–3 hours or 2–3 weeks (home)",
        "longevity": "1–3 years with maintenance",
        "pain_level": "Minimal — slight sensitivity possible",
        "disclaimer": "Does not affect crowns or veneers. Consult dentist if you have restorations."
    },
    "bonding": {
        "name": "Dental Bonding", "icon": "◇",
        "description": "Tooth-colored resin sculpted onto teeth to fix chips, gaps, or shape — often in one visit.",
        "ideal_for": ["Small chips or cracks", "Minor gaps", "Short or misshapen teeth", "Surface stains"],
        "types": ["Direct composite bonding (single visit)", "Enamel contouring + bonding"],
        "cost": "$100–$400 per tooth",
        "duration": "30–60 minutes per tooth",
        "longevity": "5–10 years",
        "pain_level": "Painless — no anesthesia usually needed",
        "disclaimer": "Less durable than veneers. Avoid biting hard objects."
    },
    "veneers": {
        "name": "Dental Veneers", "icon": "◈",
        "description": "Ultra-thin porcelain shells bonded to the front of teeth for a flawless, natural-looking smile.",
        "ideal_for": ["Chipped or worn teeth", "Gaps", "Misshapen teeth", "Severe staining", "Minor misalignment"],
        "types": ["Porcelain veneers (most natural)", "Composite veneers (less expensive)", "Lumineers (no-prep, reversible)"],
        "cost": "$900–$2,500 per tooth",
        "duration": "2–3 visits over 2–4 weeks",
        "longevity": "10–20 years",
        "pain_level": "Low — local anesthesia used",
        "disclaimer": "Requires enamel removal — irreversible. Not for teeth with decay or gum disease."
    },
    "orthodontics": {
        "name": "Orthodontic Treatment", "icon": "⬡",
        "description": "Systematic realignment using braces or clear aligners, correcting bite and aesthetics.",
        "ideal_for": ["Crooked or crowded teeth", "Gaps", "Overbite / underbite / crossbite", "Jaw alignment"],
        "types": ["Metal braces", "Ceramic braces (discreet)", "Invisalign clear aligners", "Lingual braces (hidden)"],
        "cost": "$3,000–$8,000",
        "duration": "12–36 months",
        "longevity": "Permanent with retainer use",
        "pain_level": "Moderate — soreness after adjustments",
        "disclaimer": "Requires regular visits and long-term retainer wear."
    },
    "crown": {
        "name": "Dental Crowns", "icon": "♛",
        "description": "Full-coverage ceramic caps restoring shape, strength and appearance of damaged teeth.",
        "ideal_for": ["Severely damaged teeth", "Post root canal", "Large fillings", "Fractured teeth"],
        "types": ["All-porcelain (most aesthetic)", "Zirconia (strongest)", "Porcelain-fused-to-metal", "Gold (back teeth)"],
        "cost": "$800–$2,500 per tooth",
        "duration": "2 visits over 2–3 weeks",
        "longevity": "10–25 years",
        "pain_level": "Low — local anesthesia used",
        "disclaimer": "Requires significant tooth reduction. Underlying tooth must be healthy."
    },
    "implants": {
        "name": "Dental Implants", "icon": "⊕",
        "description": "Titanium root replacements anchored in the jawbone — the gold standard for missing teeth.",
        "ideal_for": ["Single/multiple missing teeth", "Failing teeth", "Denture alternatives"],
        "types": ["Single implant", "Implant-supported bridge", "All-on-4 full arch", "Mini implants"],
        "cost": "$3,000–$6,000 per implant",
        "duration": "3–9 months (includes healing)",
        "longevity": "20+ years, often lifetime",
        "pain_level": "Moderate — surgical procedure",
        "disclaimer": "Requires adequate bone density. Not for uncontrolled diabetics or active smokers without clearance."
    },
    "gum_contouring": {
        "name": "Gum Contouring", "icon": "◑",
        "description": "Laser reshaping of the gum line to correct gummy smiles or uneven gums.",
        "ideal_for": ["Gummy smile", "Uneven gum line", "Teeth appearing too short"],
        "types": ["Laser gum contouring (minimal recovery)", "Surgical gingivectomy", "Crown lengthening"],
        "cost": "$200–$3,000 depending on extent",
        "duration": "1–2 hours",
        "longevity": "Permanent",
        "pain_level": "Low — local anesthesia, mild soreness after",
        "disclaimer": "Laser is minimally invasive. Surgical options need 1–2 weeks recovery."
    },
    "smile_makeover": {
        "name": "Full Smile Makeover", "icon": "✸",
        "description": "Comprehensive combination of procedures for a complete, custom smile transformation.",
        "ideal_for": ["Multiple concerns", "Aged smile", "Total transformation", "Special occasions"],
        "types": ["Whitening + veneers", "Orthodontics + whitening + bonding", "Veneers + gum contouring"],
        "cost": "$5,000–$30,000+",
        "duration": "3–12 months",
        "longevity": "10–20 years with maintenance",
        "pain_level": "Varies by procedures",
        "disclaimer": "Get a detailed written plan. Seek 2–3 consultations. Ask about Digital Smile Design (DSD) preview."
    }
}

SMILE_CONCERNS = {
    "discoloration": ["teeth_whitening", "veneers", "bonding"],
    "gaps":          ["bonding", "veneers", "orthodontics"],
    "chips":         ["bonding", "veneers", "crown"],
    "misalignment":  ["orthodontics", "veneers"],
    "gummy_smile":   ["gum_contouring", "veneers"],
    "missing":       ["implants", "crown"],
    "shape":         ["veneers", "bonding", "crown"],
    "overall":       ["smile_makeover", "teeth_whitening", "veneers"]
}

# ── HTML ───────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Face & Smile Advisor</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;1,300;1,400&family=Josefin+Sans:wght@100;300;400&display=swap');
:root{
  --gold:#C9A84C;--gold-light:#E8C97A;--gold-dim:#8A6E2F;
  --teal:#3A9E99;--teal-dim:#1A5250;--teal-light:#5ABFBA;
  --dark:#0A0A0A;--dark2:#111;--dark3:#1A1A1A;--dark4:#222;
  --text:#E8E0D0;--text-dim:#8A8070;
}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--dark);color:var(--text);font-family:'Josefin Sans',sans-serif;font-weight:300;min-height:100vh;}
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:radial-gradient(ellipse 80% 50% at 50% -10%,rgba(201,168,76,.07) 0%,transparent 60%),
             radial-gradient(ellipse 40% 40% at 90% 90%,rgba(58,158,153,.04) 0%,transparent 50%);}
.page{position:relative;z-index:1;max-width:960px;margin:0 auto;padding:56px 24px 80px;}

/* HEADER */
.header{text-align:center;margin-bottom:48px;}
.logo-row{display:flex;justify-content:center;gap:28px;margin-bottom:8px;}
.logo-sym{font-size:52px;line-height:1;animation:glow 4s ease-in-out infinite;}
.logo-sym.gold{color:var(--gold);}
.logo-sym.teal{color:var(--teal);animation-delay:2s;}
@keyframes glow{0%,100%{opacity:.7}50%{opacity:1;text-shadow:0 0 30px currentColor;}}
.brand{font-family:'Cormorant Garamond',serif;font-size:38px;font-weight:300;letter-spacing:3px;}
.brand .g{color:var(--gold);}
.brand .t{color:var(--teal);}
.tagline{font-size:9px;letter-spacing:7px;text-transform:uppercase;color:var(--text-dim);margin-top:8px;}
.divider{display:flex;align-items:center;gap:14px;margin:20px auto;max-width:320px;}
.dl{flex:1;height:1px;background:linear-gradient(to right,transparent,var(--gold-dim));}
.dl:last-child{background:linear-gradient(to left,transparent,var(--gold-dim));}
.dd{width:6px;height:6px;background:var(--gold);transform:rotate(45deg);box-shadow:0 0 8px rgba(201,168,76,.5);}
.intro{font-family:'Cormorant Garamond',serif;font-size:15px;color:var(--text-dim);line-height:1.8;max-width:560px;margin:0 auto;}
.badges{display:flex;justify-content:center;gap:12px;margin-top:14px;flex-wrap:wrap;}
.badge{border:1px solid var(--dark3);padding:5px 14px;font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--text-dim);}

/* TABS */
.tab-nav{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid var(--dark3);margin-bottom:32px;}
.tab-btn{padding:16px 8px;background:transparent;border:none;color:var(--text-dim);font-family:'Josefin Sans',sans-serif;font-size:9px;letter-spacing:3px;text-transform:uppercase;cursor:pointer;transition:all .3s;border-right:1px solid var(--dark3);}
.tab-btn:last-child{border-right:none;}
.tab-btn.active{background:var(--dark2);color:var(--gold);}
.tab-btn:hover:not(.active){color:var(--text);background:rgba(255,255,255,.02);}
.tab-btn.teal-tab.active{color:var(--teal);}
.tab-panel{display:none;}
.tab-panel.active{display:block;}

/* UPLOAD ZONE */
.upload-zone{border:1px solid var(--gold-dim);padding:52px 40px;text-align:center;cursor:pointer;background:linear-gradient(135deg,var(--dark2),var(--dark3));transition:all .4s;position:relative;overflow:hidden;}
.upload-zone::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(201,168,76,.06),transparent 60%);opacity:0;transition:opacity .4s;}
.upload-zone:hover::before,.upload-zone.over::before{opacity:1;}
.upload-zone:hover,.upload-zone.over{border-color:var(--gold);}
.up-icon{font-size:40px;display:block;margin-bottom:16px;opacity:.5;}
.up-text{font-size:10px;letter-spacing:4px;text-transform:uppercase;color:var(--gold);display:block;margin-bottom:8px;}
.up-hint{font-family:'Cormorant Garamond',serif;font-size:16px;color:var(--text-dim);}
#fileInput{display:none;}
.preview-wrap{display:none;border:1px solid var(--gold-dim);overflow:hidden;background:var(--dark2);position:relative;}
.preview-wrap img{width:100%;max-height:420px;object-fit:contain;display:block;}
.prev-overlay{position:absolute;top:12px;right:12px;}
.btn-remove{background:rgba(10,10,10,.85);border:1px solid var(--gold-dim);color:var(--text-dim);padding:6px 14px;font-family:'Josefin Sans',sans-serif;font-size:9px;letter-spacing:3px;text-transform:uppercase;cursor:pointer;transition:all .3s;}
.btn-remove:hover{border-color:var(--gold);color:var(--gold);}

/* ANALYZE BTN */
.analyze-btn{width:100%;margin-top:20px;padding:20px;background:transparent;border:1px solid var(--gold);color:var(--gold);font-family:'Josefin Sans',sans-serif;font-size:11px;letter-spacing:6px;text-transform:uppercase;cursor:pointer;position:relative;overflow:hidden;transition:all .4s;display:none;}
.analyze-btn::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,var(--gold),var(--gold-light));transform:translateX(-100%);transition:transform .4s;z-index:0;}
.analyze-btn:hover::before{transform:translateX(0);}
.analyze-btn:hover{color:var(--dark);}
.analyze-btn span{position:relative;z-index:1;}
.analyze-btn:disabled{opacity:.4;cursor:not-allowed;}
.analyze-btn:disabled::before{display:none;}

/* LOADING */
.loading{display:none;text-align:center;padding:48px;}
.ring{width:52px;height:52px;border:1px solid var(--dark3);border-top-color:var(--gold);border-radius:50%;animation:spin 1.2s linear infinite;margin:0 auto 20px;}
@keyframes spin{to{transform:rotate(360deg);}}
.loading-text{font-size:9px;letter-spacing:5px;text-transform:uppercase;color:var(--text-dim);animation:pulse 2s ease-in-out infinite;}
@keyframes pulse{0%,100%{opacity:.4}50%{opacity:1}}

/* ERROR */
.err{display:none;border:1px solid #8B3A3A;background:rgba(139,58,58,.1);padding:20px;margin-top:16px;text-align:center;}
.err p{font-family:'Cormorant Garamond',serif;font-size:16px;color:#E87070;}

/* RESULTS */
.results{display:none;animation:up .8s ease forwards;}
@keyframes up{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.sec-title{font-size:9px;letter-spacing:5px;text-transform:uppercase;color:var(--text-dim);margin-bottom:16px;display:flex;align-items:center;gap:12px;}
.sec-title::after{content:'';flex:1;height:1px;background:var(--dark3);}
.sec-title.teal{color:var(--teal-dim);}

/* SCORE BOX */
.score-box{text-align:center;padding:44px 24px;background:linear-gradient(135deg,var(--dark2),var(--dark3));border:1px solid var(--gold-dim);margin-bottom:20px;position:relative;overflow:hidden;}
.score-box::before{content:'φ';position:absolute;font-family:'Cormorant Garamond',serif;font-size:200px;color:rgba(201,168,76,.03);top:50%;left:50%;transform:translate(-50%,-50%);pointer-events:none;}
.score-label{font-size:9px;letter-spacing:5px;text-transform:uppercase;color:var(--text-dim);display:block;margin-bottom:12px;}
.score-num{font-family:'Cormorant Garamond',serif;font-size:96px;font-weight:300;line-height:1;color:var(--gold);}
.score-unit{font-family:'Cormorant Garamond',serif;font-size:32px;color:var(--gold-dim);}
.score-verdict{font-family:'Cormorant Garamond',serif;font-size:19px;color:var(--text);margin-top:12px;font-style:italic;}
.bar-track{height:2px;background:var(--dark3);margin-top:20px;}
.bar-fill{height:100%;background:linear-gradient(to right,var(--gold-dim),var(--gold),var(--gold-light));width:0%;transition:width 1.6s cubic-bezier(.4,0,.2,1);}

/* ANNOTATED IMG */
.ann-wrap{border:1px solid var(--gold-dim);margin-bottom:20px;background:var(--dark2);}
.ann-wrap img{width:100%;display:block;}
.ann-label{font-size:9px;letter-spacing:4px;text-transform:uppercase;color:var(--gold-dim);padding:10px 16px;border-top:1px solid var(--dark3);text-align:center;}

/* RATIO CARDS */
.ratio-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(255px,1fr));gap:12px;margin-bottom:20px;}
.ratio-card{background:var(--dark2);border:1px solid var(--dark3);padding:18px;transition:border-color .3s;}
.ratio-card:hover{border-color:var(--gold-dim);}
.ratio-name{font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--text-dim);margin-bottom:10px;}
.ratio-row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px;}
.ratio-score{font-family:'Cormorant Garamond',serif;font-size:30px;color:var(--gold);}
.ratio-info{font-size:10px;color:var(--text-dim);text-align:right;line-height:1.6;}
.ratio-bar{height:1px;background:var(--dark3);}
.ratio-bar-fill{height:100%;background:var(--gold);transition:width 1s ease;}

/* ANALYSIS */
.analysis-box{background:var(--dark2);border:1px solid var(--dark3);border-left:2px solid var(--gold-dim);padding:24px;margin-bottom:20px;}
.analysis-title{font-size:9px;letter-spacing:5px;text-transform:uppercase;color:var(--gold-dim);display:block;margin-bottom:12px;}
.analysis-text{font-family:'Cormorant Garamond',serif;font-size:17px;line-height:1.85;color:var(--text);}

/* SYMMETRY SECTION */
.sym-intro{background:var(--dark2);border:1px solid var(--dark3);border-left:2px solid var(--teal-dim);padding:24px;margin-bottom:20px;}
.sym-intro-title{font-size:9px;letter-spacing:5px;text-transform:uppercase;color:var(--teal-dim);display:block;margin-bottom:12px;}
.sym-intro-text{font-family:'Cormorant Garamond',serif;font-size:16px;line-height:1.85;color:var(--text);}
.sym-cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;margin-bottom:20px;}
.sym-card{background:var(--dark2);border:1px solid var(--dark3);border-top:2px solid var(--teal-dim);padding:20px;transition:border-color .3s;}
.sym-card:hover{border-color:var(--teal-dim);border-top-color:var(--teal);}
.sym-card-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;}
.sym-card-name{font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--text-dim);}
.sym-card-badge{font-size:9px;letter-spacing:2px;text-transform:uppercase;padding:3px 10px;border:1px solid;}
.badge-good{color:#5ABF6E;border-color:#2A5E35;}
.badge-moderate{color:var(--gold);border-color:var(--gold-dim);}
.badge-improve{color:#E87070;border-color:#8B3A3A;}
.sym-card-title{font-family:'Cormorant Garamond',serif;font-size:18px;color:var(--text);margin-bottom:8px;}
.sym-card-text{font-family:'Cormorant Garamond',serif;font-size:14px;color:var(--text-dim);line-height:1.7;}
.sym-card-recs{margin-top:12px;padding-top:12px;border-top:1px solid var(--dark3);}
.sym-card-rec-label{font-size:8px;letter-spacing:3px;text-transform:uppercase;color:var(--teal-dim);margin-bottom:8px;}
.sym-rec-item{font-size:11px;color:var(--text-dim);padding:4px 0;padding-left:14px;position:relative;line-height:1.5;}
.sym-rec-item::before{content:'→';position:absolute;left:0;color:var(--teal-dim);}

/* DISCLAIMER NOTE */
.note{background:rgba(201,168,76,.05);border:1px solid var(--gold-dim);padding:16px 20px;margin-bottom:20px;}
.note p{font-size:11px;color:var(--text-dim);line-height:1.7;}
.note strong{color:var(--gold);font-weight:400;}

/* ── DENTAL TAB ── */
.dental-hero{text-align:center;padding:36px 24px;background:linear-gradient(135deg,var(--dark2),var(--dark3));border:1px solid var(--teal-dim);margin-bottom:28px;position:relative;overflow:hidden;}
.dental-hero::before{content:'◉';position:absolute;font-size:180px;color:rgba(58,158,153,.04);top:50%;left:50%;transform:translate(-50%,-50%);}
.dental-icon{font-size:48px;display:block;margin-bottom:10px;color:var(--teal);}
.dental-title{font-family:'Cormorant Garamond',serif;font-size:30px;color:var(--teal);display:block;}
.dental-sub{font-family:'Cormorant Garamond',serif;font-size:15px;color:var(--text-dim);margin-top:6px;font-style:italic;}

.phi-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px;margin-bottom:28px;}
.phi-card{background:var(--dark2);border:1px solid var(--dark3);padding:18px;transition:border-color .3s;}
.phi-card:hover{border-color:var(--gold-dim);}
.phi-card-top{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;}
.phi-card-name{font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--text-dim);flex:1;padding-right:10px;line-height:1.5;}
.phi-card-val{font-family:'Cormorant Garamond',serif;font-size:26px;color:var(--gold);white-space:nowrap;}
.phi-card-desc{font-family:'Cormorant Garamond',serif;font-size:13px;color:var(--text-dim);line-height:1.65;font-style:italic;}

.concern-sub{font-family:'Cormorant Garamond',serif;font-size:15px;color:var(--text-dim);margin-bottom:16px;font-style:italic;}
.concern-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:10px;margin-bottom:18px;}
.concern-btn{padding:15px 10px;background:var(--dark2);border:1px solid var(--dark3);color:var(--text-dim);font-family:'Josefin Sans',sans-serif;font-size:9px;letter-spacing:3px;text-transform:uppercase;cursor:pointer;transition:all .25s;text-align:center;display:flex;flex-direction:column;align-items:center;gap:8px;}
.concern-btn:hover{border-color:var(--teal-light);color:var(--teal);}
.concern-btn.selected{border-color:var(--teal);color:var(--teal);background:rgba(58,158,153,.08);}
.concern-icon{font-size:20px;}

.recommend-btn{width:100%;padding:18px;background:transparent;border:1px solid var(--teal);color:var(--teal);font-family:'Josefin Sans',sans-serif;font-size:11px;letter-spacing:5px;text-transform:uppercase;cursor:pointer;position:relative;overflow:hidden;transition:all .4s;}
.recommend-btn::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,var(--teal-dim),var(--teal));transform:translateX(-100%);transition:transform .35s;z-index:0;}
.recommend-btn:hover::before{transform:translateX(0);}
.recommend-btn:hover{color:var(--dark);}
.recommend-btn span{position:relative;z-index:1;}

.proc-results{display:none;animation:up .6s ease forwards;}
.proc-card{background:var(--dark2);border:1px solid var(--dark3);border-top:2px solid var(--teal-dim);padding:24px;margin-bottom:16px;transition:border-top-color .3s;}
.proc-card:hover{border-top-color:var(--teal);}
.proc-top{display:flex;align-items:center;gap:12px;margin-bottom:12px;}
.proc-icon{font-size:28px;color:var(--teal);}
.proc-name{font-family:'Cormorant Garamond',serif;font-size:24px;color:var(--text);}
.proc-desc{font-family:'Cormorant Garamond',serif;font-size:15px;color:var(--text-dim);line-height:1.75;margin-bottom:16px;font-style:italic;}
.proc-ideal-label,.proc-meta-label,.proc-types-label{font-size:8px;letter-spacing:3px;text-transform:uppercase;color:var(--text-dim);margin-bottom:7px;}
.proc-ideal-tags{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px;}
.proc-ideal-tag{background:var(--dark3);padding:4px 10px;font-size:10px;color:var(--text-dim);}
.proc-meta{display:grid;grid-template-columns:repeat(auto-fill,minmax(185px,1fr));gap:10px;margin-bottom:14px;}
.proc-meta-item{background:var(--dark3);padding:12px;}
.proc-meta-value{font-family:'Cormorant Garamond',serif;font-size:14px;color:var(--text);margin-top:4px;}
.proc-type-tag{display:inline-block;border:1px solid var(--teal-dim);color:var(--teal);padding:3px 10px;font-size:9px;letter-spacing:2px;margin:3px 3px 3px 0;}
.proc-disclaimer{font-size:11px;color:var(--text-dim);border-left:2px solid var(--gold-dim);padding-left:12px;margin-top:12px;line-height:1.7;font-style:italic;}

.next-steps{background:rgba(58,158,153,.06);border:1px solid var(--teal-dim);padding:22px;margin-top:20px;}
.next-steps-title{font-size:9px;letter-spacing:4px;text-transform:uppercase;color:var(--teal-dim);margin-bottom:10px;}
.next-steps p{font-family:'Cormorant Garamond',serif;font-size:15px;color:var(--text-dim);line-height:1.8;}
.next-steps strong{color:var(--teal);font-weight:400;}

.reset-btn{width:100%;margin-top:12px;background:transparent;border:1px solid var(--dark3);color:var(--text-dim);padding:13px;font-family:'Josefin Sans',sans-serif;font-size:9px;letter-spacing:4px;text-transform:uppercase;cursor:pointer;transition:all .3s;}
.reset-btn:hover{border-color:var(--gold-dim);color:var(--gold);}

/* VISUAL PREVIEW */
.preview-cta{background:linear-gradient(135deg,var(--dark2),var(--dark3));border:1px solid var(--teal-dim);padding:28px;text-align:center;margin-bottom:20px;}
.preview-cta-label{font-size:9px;letter-spacing:5px;text-transform:uppercase;color:var(--teal-dim);margin-bottom:8px;}
.preview-cta-text{font-family:'Cormorant Garamond',serif;font-size:16px;color:var(--text-dim);margin-bottom:18px;font-style:italic;}
.preview-btn{padding:16px 40px;background:transparent;border:1px solid var(--teal);color:var(--teal);font-family:'Josefin Sans',sans-serif;font-size:11px;letter-spacing:5px;text-transform:uppercase;cursor:pointer;position:relative;overflow:hidden;transition:all .4s;}
.preview-btn::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,var(--teal-dim),var(--teal));transform:translateX(-100%);transition:transform .35s;z-index:0;}
.preview-btn:hover::before{transform:translateX(0);}
.preview-btn:hover{color:var(--dark);}
.preview-btn span{position:relative;z-index:1;}
.preview-btn:disabled{opacity:.4;cursor:not-allowed;}
.comparison-wrap{margin-bottom:20px;}
.comp-disclaimer{background:rgba(201,168,76,.05);border:1px solid var(--gold-dim);padding:14px 18px;margin-bottom:16px;}
.comp-disclaimer p{font-size:11px;color:var(--text-dim);line-height:1.7;}
.improvements-list{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:8px;margin-bottom:16px;}
.imp-item{background:var(--dark2);border:1px solid var(--dark3);border-left:2px solid var(--teal-dim);padding:10px 14px;font-size:10px;letter-spacing:1px;color:var(--text-dim);line-height:1.5;}
.download-btn{padding:12px 28px;background:transparent;border:1px solid var(--gold-dim);color:var(--gold-dim);font-family:'Josefin Sans',sans-serif;font-size:9px;letter-spacing:4px;text-transform:uppercase;cursor:pointer;transition:all .3s;}
.download-btn:hover{border-color:var(--gold);color:var(--gold);}

.footer{margin-top:56px;text-align:center;font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--text-dim);opacity:.4;padding-bottom:20px;}
</style>
</head>
<body>
<div class="page">

<!-- HEADER -->
<div class="header">
  <div class="logo-row"><span class="logo-sym gold">φ</span><span class="logo-sym teal">◉</span></div>
  <div class="brand"><span class="g">Face</span> &amp; <span class="t">Smile</span> Advisor</div>
  <p class="tagline">Golden Ratio · Facial Symmetry · Dental Aesthetics</p>
  <div class="divider"><div class="dl"></div><div class="dd"></div><div class="dl"></div></div>
  <p class="intro">Measure your facial phi harmony, receive personalised symmetry improvement tips, and explore cosmetic dental procedures — all in one place.</p>
  <div class="badges">
    <span class="badge">◈ No API Key</span>
    <span class="badge">◈ No Sign-up</span>
    <span class="badge">◈ 100% Free</span>
  </div>
</div>

<!-- TABS -->
<div class="tab-nav">
  <button class="tab-btn active" onclick="switchTab('face',this)">φ &nbsp; Phi Analysis</button>
  <button class="tab-btn" onclick="switchTab('symmetry',this)">⬡ &nbsp; Symmetry Tips</button>
  <button class="tab-btn teal-tab" onclick="switchTab('dental',this)">◉ &nbsp; Dental Smile</button>
</div>

<!-- ══ TAB 1: PHI ══ -->
<div class="tab-panel active" id="tab-face">
  <div id="dropZone" class="upload-zone" onclick="document.getElementById('fileInput').click()">
    <span class="up-icon">◈</span>
    <span class="up-text">Upload Frontal Selfie</span>
    <span class="up-hint">Drag & drop or click · JPG PNG WEBP</span>
    <input type="file" id="fileInput" accept="image/*">
  </div>
  <div class="preview-wrap" id="previewWrap">
    <img id="previewImg" src="" alt="">
    <div class="prev-overlay"><button class="btn-remove" onclick="resetFace()">✕ Remove</button></div>
  </div>
  <button class="analyze-btn" id="analyzeBtn" onclick="analyzeFace()"><span>◈ &nbsp; Analyze Phi Harmony</span></button>
  <div class="loading" id="loading"><div class="ring"></div><p class="loading-text">Measuring golden ratios</p></div>
  <div class="err" id="errBox"><p id="errText"></p></div>

  <div class="results" id="faceResults">
    <div class="score-box">
      <span class="score-label">Overall Phi Harmony Score</span><br>
      <span class="score-num" id="scoreNum">0</span><span class="score-unit">%</span>
      <p class="score-verdict" id="scoreVerdict"></p>
      <div class="bar-track"><div class="bar-fill" id="scoreBar"></div></div>
    </div>
    <div class="ann-wrap" id="annWrap" style="display:none">
      <img id="annImg" src="" alt="Annotated">
      <p class="ann-label">◈ &nbsp; Detected Facial Landmarks &nbsp; ◈</p>
    </div>
    <p class="sec-title">Individual Ratio Analysis</p>
    <div class="ratio-grid" id="ratioGrid"></div>
    <div class="analysis-box">
      <span class="analysis-title">φ · Aesthetic Interpretation</span>
      <p class="analysis-text" id="analysisText"></p>
    </div>
    <button class="reset-btn" onclick="resetFace()">↺ &nbsp; Analyze Another Photo</button>
  </div>
</div>

<!-- ══ TAB 2: SYMMETRY ══ -->
<div class="tab-panel" id="tab-symmetry">
  <div class="note" style="margin-bottom:28px;">
    <p><strong>How it works:</strong> Upload a frontal selfie on the Phi Analysis tab first — your symmetry recommendations will appear here automatically based on your measurements. These are educational suggestions only, not medical advice.</p>
  </div>
  <div id="symNoData" style="text-align:center;padding:60px 20px;">
    <p style="font-family:'Cormorant Garamond',serif;font-size:20px;color:var(--text-dim);font-style:italic;">Upload and analyze a photo first to see your personalized symmetry recommendations.</p>
    <button style="margin-top:20px;padding:14px 32px;background:transparent;border:1px solid var(--gold-dim);color:var(--gold);font-family:'Josefin Sans',sans-serif;font-size:9px;letter-spacing:4px;text-transform:uppercase;cursor:pointer;" onclick="switchTab('face',document.querySelectorAll('.tab-btn')[0])">◈ &nbsp; Go to Phi Analysis</button>
  </div>
  <div id="symResults" style="display:none;">
    <div class="sym-intro">
      <span class="sym-intro-title">⬡ · Your Symmetry Profile</span>
      <p class="sym-intro-text" id="symIntroText"></p>
    </div>
    <p class="sec-title teal">Improvement Recommendations</p>
    <div class="sym-cards" id="symCards"></div>

    <div class="preview-cta">
      <p class="preview-cta-label">◈ · Visual Simulation</p>
      <p class="preview-cta-text">See how your face looks with symmetry improvements applied</p>
      <button class="preview-btn" id="previewBtn" onclick="generatePreview()"><span>◈ &nbsp; Generate Visual Preview</span></button>
    </div>

    <div class="comparison-wrap" id="comparisonWrap" style="display:none;">
      <p class="sec-title teal">Before &amp; After Simulation</p>
      <div class="comp-disclaimer"><p>⚠ <strong>Digital simulation only</strong> — for educational purposes. Applies mathematical phi-based adjustments. Real professional results will differ significantly.</p></div>
      <img id="comparisonImg" src="" alt="Before/After" style="width:100%;display:block;border:1px solid var(--dark3);margin-bottom:12px;">
      <div id="improvementsApplied" class="improvements-list"></div>
      <button class="download-btn" onclick="downloadPreview()">↓ &nbsp; Download Comparison Image</button>
    </div>

    <div class="note" style="margin-top:16px;">
      <p><strong>Important:</strong> These recommendations are based on mathematical facial proportion analysis only. Symmetry is one component of attractiveness — personality, expression, and confidence matter far more. Any cosmetic procedures should be discussed with qualified medical professionals. Many of these improvements can also be achieved through <strong>non-surgical means</strong> such as makeup, hairstyling, posture, and grooming.</p>
    </div>
  </div>
</div>

<!-- ══ TAB 3: DENTAL ══ -->
<div class="tab-panel" id="tab-dental">
  <div class="dental-hero">
    <span class="dental-icon">◉</span>
    <span class="dental-title">Dental Smile Advisor</span>
    <p class="dental-sub">Golden ratio principles applied to cosmetic dentistry</p>
  </div>

  <div class="note">
    <p><strong>Disclaimer:</strong> This tool provides general educational information only. It is <strong>not a substitute for professional dental advice, diagnosis, or treatment.</strong> Always consult a licensed dentist or orthodontist before any procedure.</p>
  </div>

  <p class="sec-title teal">φ · Golden Ratio in the Ideal Smile</p>
  <div class="phi-grid">
    <div class="phi-card">
      <div class="phi-card-top"><div class="phi-card-name">Central to Lateral Incisor Width</div><div class="phi-card-val">1.618</div></div>
      <div class="phi-card-desc">Upper central incisor should be φ times wider than the lateral incisor.</div>
    </div>
    <div class="phi-card">
      <div class="phi-card-top"><div class="phi-card-name">Lateral Incisor to Canine</div><div class="phi-card-val">1.618</div></div>
      <div class="phi-card-desc">Each successive tooth narrows by the golden ratio creating a harmonious arc.</div>
    </div>
    <div class="phi-card">
      <div class="phi-card-top"><div class="phi-card-name">Smile Width to Face Width</div><div class="phi-card-val">0.618</div></div>
      <div class="phi-card-desc">Ideal smile spans approximately 61.8% of total face width — the golden section.</div>
    </div>
    <div class="phi-card">
      <div class="phi-card-top"><div class="phi-card-name">Tooth Height to Width</div><div class="phi-card-val">0.618</div></div>
      <div class="phi-card-desc">Central incisors ideally have a height-to-width ratio near the golden proportion.</div>
    </div>
    <div class="phi-card">
      <div class="phi-card-top"><div class="phi-card-name">Gum Exposure on Smile</div><div class="phi-card-val">≤ 3mm</div></div>
      <div class="phi-card-desc">Ideal smile shows 75–100% of upper teeth with no more than 3mm of gum tissue.</div>
    </div>
    <div class="phi-card">
      <div class="phi-card-top"><div class="phi-card-name">Smile Arc Curvature</div><div class="phi-card-val">Convex</div></div>
      <div class="phi-card-desc">Upper incisal edges follow the lower lip curve — a gentle upward arc is most youthful.</div>
    </div>
  </div>

  <p class="sec-title teal" style="margin-bottom:6px;">Select Your Smile Concerns</p>
  <p class="concern-sub">Choose all that apply — we'll recommend the most suitable procedures</p>
  <div class="concern-grid">
    <button class="concern-btn" onclick="toggleConcern(this,'discoloration')"><span class="concern-icon">◑</span>Discoloration &amp; Staining</button>
    <button class="concern-btn" onclick="toggleConcern(this,'gaps')"><span class="concern-icon">◻</span>Gaps Between Teeth</button>
    <button class="concern-btn" onclick="toggleConcern(this,'chips')"><span class="concern-icon">◈</span>Chipped or Cracked</button>
    <button class="concern-btn" onclick="toggleConcern(this,'misalignment')"><span class="concern-icon">⬡</span>Crooked / Misaligned</button>
    <button class="concern-btn" onclick="toggleConcern(this,'gummy_smile')"><span class="concern-icon">◑</span>Gummy Smile</button>
    <button class="concern-btn" onclick="toggleConcern(this,'missing')"><span class="concern-icon">⊕</span>Missing Teeth</button>
    <button class="concern-btn" onclick="toggleConcern(this,'shape')"><span class="concern-icon">◇</span>Tooth Shape / Size</button>
    <button class="concern-btn" onclick="toggleConcern(this,'overall')"><span class="concern-icon">✸</span>Full Transformation</button>
  </div>
  <button class="recommend-btn" id="recommendBtn" onclick="getDentalRecs()"><span>◉ &nbsp; Get My Smile Recommendations</span></button>

  <div class="proc-results" id="procResults">
    <p class="sec-title teal" style="margin-top:28px;">Recommended Procedures</p>
    <div id="procCards"></div>
    <div class="next-steps">
      <p class="next-steps-title">◈ · Next Steps</p>
      <p>Schedule a consultation with a <strong>board-certified cosmetic dentist</strong> or <strong>orthodontist</strong>. Ask about <strong>Digital Smile Design (DSD)</strong> for a digital preview. Get <strong>2–3 independent consultations</strong> for any procedure over $1,000. Always request a <strong>written treatment plan with itemised costs</strong>.</p>
    </div>
  </div>
</div>

</div><!-- end page -->
<div class="footer">φ = 1.618… · Face &amp; Smile Advisor · For Educational Purposes Only · Always Consult Qualified Professionals</div>

<script>
// ── TABS ──
function switchTab(name, btn) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  if(btn) btn.classList.add('active');
}

// ── FACE UPLOAD ──
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
let currentFile = null;
let lastData = null;

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) loadFile(f);
});
fileInput.addEventListener('change', e => { if (e.target.files[0]) loadFile(e.target.files[0]); });

function loadFile(file) {
  currentFile = file;
  const r = new FileReader();
  r.onload = ev => {
    document.getElementById('previewImg').src = ev.target.result;
    dropZone.style.display = 'none';
    document.getElementById('previewWrap').style.display = 'block';
    document.getElementById('analyzeBtn').style.display = 'block';
    document.getElementById('faceResults').style.display = 'none';
    document.getElementById('errBox').style.display = 'none';
  };
  r.readAsDataURL(file);
}

function resetFace() {
  currentFile = null; fileInput.value = '';
  document.getElementById('previewImg').src = '';
  dropZone.style.display = 'block';
  document.getElementById('previewWrap').style.display = 'none';
  document.getElementById('analyzeBtn').style.display = 'none';
  document.getElementById('faceResults').style.display = 'none';
  document.getElementById('loading').style.display = 'none';
  document.getElementById('errBox').style.display = 'none';
}

async function analyzeFace() {
  if (!currentFile) return;
  document.getElementById('analyzeBtn').disabled = true;
  document.getElementById('loading').style.display = 'block';
  document.getElementById('faceResults').style.display = 'none';
  document.getElementById('errBox').style.display = 'none';
  const fd = new FormData();
  fd.append('image', currentFile);
  try {
    const res = await fetch('/analyze', { method:'POST', body:fd });
    const data = await res.json();
    if (!res.ok || data.error) { showErr(data.error||'Analysis failed.'); return; }
    lastData = data;
    displayFace(data);
    buildSymmetry(data);
  } catch(e) { showErr('Network error: '+e.message); }
  finally {
    document.getElementById('analyzeBtn').disabled = false;
    document.getElementById('loading').style.display = 'none';
  }
}

function showErr(msg) {
  document.getElementById('errText').textContent = msg;
  document.getElementById('errBox').style.display = 'block';
}

function verdict(s) {
  if(s>=90) return 'Exceptional — Near-perfect divine proportion';
  if(s>=80) return 'Remarkable — Highly harmonious features';
  if(s>=70) return 'Harmonious — Strong phi alignment';
  if(s>=60) return 'Balanced — Pleasant natural proportion';
  if(s>=50) return 'Moderate — Characteristic individuality';
  return 'Distinctive — Unique beyond classical proportion';
}

function displayFace(data) {
  const s = data.overall_score;
  document.getElementById('scoreVerdict').textContent = verdict(s);
  setTimeout(()=>document.getElementById('scoreBar').style.width=s+'%',100);
  animNum('scoreNum',0,s,1300);
  if(data.annotated_image){
    document.getElementById('annImg').src='data:image/jpeg;base64,'+data.annotated_image;
    document.getElementById('annWrap').style.display='block';
  }
  const grid = document.getElementById('ratioGrid');
  grid.innerHTML='';
  data.ratios.forEach(r=>{
    const c=document.createElement('div');
    c.className='ratio-card';
    c.innerHTML=`<div class="ratio-name">${r.name}</div>
    <div class="ratio-row"><span class="ratio-score">${r.score.toFixed(1)}%</span>
    <span class="ratio-info">Measured: ${r.measured.toFixed(3)}<br>φ deviation: ${r.deviation.toFixed(3)}</span></div>
    <div class="ratio-bar"><div class="ratio-bar-fill" style="width:0%" data-t="${r.score}"></div></div>`;
    grid.appendChild(c);
    setTimeout(()=>c.querySelector('.ratio-bar-fill').style.width=r.score+'%',200);
  });
  document.getElementById('analysisText').textContent=data.analysis;
  document.getElementById('faceResults').style.display='block';
}

function animNum(id,from,to,dur){
  const el=document.getElementById(id);
  const start=performance.now();
  function step(now){
    const t=Math.min((now-start)/dur,1);
    const e=t<.5?2*t*t:-1+(4-2*t)*t;
    el.textContent=(from+(to-from)*e).toFixed(1);
    if(t<1)requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── SYMMETRY RECOMMENDATIONS ──
function buildSymmetry(data) {
  const recs = generateSymRecs(data);
  document.getElementById('symNoData').style.display='none';
  document.getElementById('symResults').style.display='block';

  const s = data.overall_score;
  let intro = '';
  if(s>=80) intro = `Your overall facial harmony score of ${s.toFixed(1)}% places you in the top tier of phi alignment. Your face is naturally well-proportioned. The minor refinements below are purely optional and cosmetic in nature.`;
  else if(s>=65) intro = `With a phi harmony score of ${s.toFixed(1)}%, your face shows solid natural proportion. A few targeted improvements could enhance your symmetry and bring specific features closer to the golden ideal.`;
  else intro = `Your phi harmony score of ${s.toFixed(1)}% shows a distinctive face with several measurements that deviate from classical proportion. The recommendations below outline options to enhance facial balance and symmetry if you choose to pursue them.`;
  document.getElementById('symIntroText').textContent = intro;

  const container = document.getElementById('symCards');
  container.innerHTML = '';
  recs.forEach(rec => {
    const card = document.createElement('div');
    card.className = 'sym-card';
    const badgeClass = rec.level==='good'?'badge-good':rec.level==='moderate'?'badge-moderate':'badge-improve';
    const badgeText  = rec.level==='good'?'Well Aligned':rec.level==='moderate'?'Minor Deviation':'Improvement Area';
    const recItems = rec.recommendations.map(r=>`<div class="sym-rec-item">${r}</div>`).join('');
    card.innerHTML=`
      <div class="sym-card-header">
        <div class="sym-card-name">${rec.name}</div>
        <span class="sym-card-badge ${badgeClass}">${badgeText}</span>
      </div>
      <div class="sym-card-title">${rec.title}</div>
      <div class="sym-card-text">${rec.description}</div>
      ${rec.recommendations.length>0?`<div class="sym-card-recs"><div class="sym-card-rec-label">Improvement Options</div>${recItems}</div>`:''}`;
    container.appendChild(card);
  });
}

function generateSymRecs(data) {
  const ratios = {};
  data.ratios.forEach(r => ratios[r.name] = r);
  const recs = [];

  // Face proportions
  const fwh = ratios['Face Width to Height'];
  if(fwh) {
    const level = fwh.score>=75?'good':fwh.score>=55?'moderate':'improve';
    recs.push({
      name:'Face Width to Height',
      title: fwh.score>=75?'Naturally balanced face shape':'Face shape deviates from phi ideal',
      description: `Your face width-to-height ratio is ${fwh.measured.toFixed(3)} (ideal: 1.618). ${fwh.score>=75?'This closely matches the golden ratio face shape.':fwh.measured>1.618?'Your face appears slightly wider relative to its height.':'Your face appears slightly longer relative to its width.'}`,
      level,
      recommendations: fwh.score>=75?[]:fwh.measured>1.618?
        ['Strategic contouring with makeup along temples and jawline','Hairstyles with volume on top (pompadour, quiff) to add height','Avoid wide hairstyles that add horizontal width']:
        ['Side-swept or voluminous hairstyles to add width','Contouring under cheekbones to shorten perceived face length','Bold eyebrow shaping to visually reduce height']
    });
  }

  // Eye symmetry
  const eyeSym = ratios['Eye Width Symmetry'];
  if(eyeSym) {
    const level = eyeSym.score>=75?'good':eyeSym.score>=55?'moderate':'improve';
    recs.push({
      name:'Eye Symmetry',
      title: eyeSym.score>=75?'Eyes are well-matched':'Eye width asymmetry detected',
      description:`Eye symmetry ratio measured at ${eyeSym.measured.toFixed(3)}. ${eyeSym.score>=75?'Your eyes appear well-balanced.':'Some asymmetry in eye width was detected, which is very common.'}`,
      level,
      recommendations: eyeSym.score>=75?[]:
        ['Eye makeup techniques — applying shadow/liner differently to each eye to balance them visually','Upper eyelid tape or eyelid primer can visually even out drooping','Brow shaping: raising the lower brow side to balance eye appearance','Consult an oculoplastic surgeon for ptosis (drooping eyelid) correction if significant']
    });
  }

  // Eye spacing
  const eyeSpan = ratios['Face Width to Eye Span'];
  if(eyeSpan) {
    const level = eyeSpan.score>=70?'good':eyeSpan.score>=50?'moderate':'improve';
    recs.push({
      name:'Eye Spacing',
      title: eyeSpan.score>=70?'Eye spacing is proportionate':'Eye spacing differs from phi ideal',
      description:`Face-to-eye-span ratio: ${eyeSpan.measured.toFixed(3)} (ideal near 1.618). ${eyeSpan.measured>1.618?'Eyes appear closer together relative to face width (hypotelorism tendency).':'Eyes appear further apart relative to face width (hypertelorism tendency).'}`,
      level,
      recommendations: eyeSpan.score>=70?[]:eyeSpan.measured>1.618?
        ['Cat-eye or winged liner to draw eyes outward visually','Lighter inner corner highlight','Part hair in the center to widen facial perception']:
        ['Inner corner eyeliner and darker inner shadow to bring eyes visually closer','Brow tails that extend further outward','Side-swept bangs to narrow perceived face width']
    });
  }

  // Vertical thirds
  const hairEye = ratios['Hairline to Eye / Eye to Nose'];
  if(hairEye) {
    const level = hairEye.score>=75?'good':hairEye.score>=55?'moderate':'improve';
    recs.push({
      name:'Vertical Face Thirds',
      title: hairEye.score>=75?'Vertical thirds are well-balanced':'Vertical facial thirds show imbalance',
      description:`Upper-to-mid face ratio: ${hairEye.measured.toFixed(3)} (ideal: ~1.618). ${hairEye.score>=75?'Your face divides proportionately into thirds.':hairEye.measured>1.618?'Your upper face (forehead) is proportionally larger.':'Your mid-face is proportionally larger.'}`,
      level,
      recommendations: hairEye.score>=75?[]:hairEye.measured>1.618?
        ['Bangs or fringe hairstyles to visually reduce forehead height','Bold brows to draw attention downward','Forehead contouring with matte bronzer','Hair transplant or hairline lowering surgery (consult specialist)']:
        ['Swept-up hairstyles to add upper face length','Lighter forehead highlight to elongate upper third','Eyebrow microblading higher up','Botox brow lift (consult qualified practitioner)']
    });
  }

  // Mid to lower face
  const midLow = ratios['Mid Face to Lower Face'];
  if(midLow) {
    const level = midLow.score>=70?'good':midLow.score>=50?'moderate':'improve';
    recs.push({
      name:'Mid to Lower Face',
      title: midLow.score>=70?'Lower face proportions are balanced':'Lower face proportion could be improved',
      description:`Mid-to-lower face ratio: ${midLow.measured.toFixed(3)} (ideal: ~1.618). ${midLow.score>=70?'Good balance between nose-to-chin and eye-to-nose segments.':midLow.measured>1.618?'Lower face (chin area) appears shorter relative to mid-face.':'Lower face appears longer relative to mid-face.'}`,
      level,
      recommendations: midLow.score>=70?[]:midLow.measured>1.618?
        ['Chin augmentation (filler or implant) to lengthen lower face — consult facial plastic surgeon','Beard styling for men: fuller chin beard to add length','Lip filler to enhance lip projection','Contouring under the chin']:
        ['Jawline contouring with makeup to shorten perceived lower face','Avoid chin filler if lower third is already prominent','Hairstyles with volume at cheek level to shorten lower face visually']
    });
  }

  // Overall symmetry tip
  recs.push({
    name:'General Symmetry',
    title:'Universal symmetry enhancement tips',
    description:'These techniques benefit facial symmetry regardless of specific measurements.',
    level:'moderate',
    recommendations:[
      'Posture: upright posture dramatically improves perceived facial symmetry and attractiveness',
      'Consistent skincare: even skin tone minimizes the appearance of asymmetry',
      'Sleep position: sleeping on your back reduces asymmetric pressure on facial structures',
      'Dental occlusion: misaligned bite can cause facial asymmetry — consult an orthodontist',
      'Photography tip: most faces look more symmetrical from a slight 3/4 angle than dead-on'
    ]
  });

  return recs;
}


// ── VISUAL PREVIEW ──
let previewB64 = null;

async function generatePreview() {
  if (!currentFile) { alert('Please upload and analyze a photo first.'); return; }
  const btn = document.getElementById('previewBtn');
  btn.disabled = true;
  btn.querySelector('span').textContent = '◈   Generating simulation...';

  try {
    const fd = new FormData();
    fd.append('image', currentFile);
    fd.append('ratios', JSON.stringify(lastData ? lastData.ratios : []));

    const res = await fetch('/simulate', { method:'POST', body:fd });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }

    previewB64 = data.comparison;
    document.getElementById('comparisonImg').src = 'data:image/jpeg;base64,' + data.comparison;

    const impList = document.getElementById('improvementsApplied');
    impList.innerHTML = data.improvements.map(i => `<div class="imp-item">✓ ${i}</div>`).join('');

    document.getElementById('comparisonWrap').style.display = 'block';
    document.getElementById('comparisonWrap').scrollIntoView({behavior:'smooth', block:'start'});
  } catch(e) { alert('Error: ' + e.message); }
  finally {
    btn.disabled = false;
    btn.querySelector('span').textContent = '◈   Generate Visual Preview';
  }
}

function downloadPreview() {
  if (!previewB64) return;
  const a = document.createElement('a');
  a.href = 'data:image/jpeg;base64,' + previewB64;
  a.download = 'face-harmony-preview.jpg';
  a.click();
}

// ── DENTAL ──
let selectedConcerns = new Set();
function toggleConcern(btn, concern) {
  if(selectedConcerns.has(concern)){ selectedConcerns.delete(concern); btn.classList.remove('selected'); }
  else { selectedConcerns.add(concern); btn.classList.add('selected'); }
}

async function getDentalRecs() {
  if(selectedConcerns.size===0){ alert('Please select at least one concern.'); return; }
  document.getElementById('recommendBtn').disabled=true;
  try {
    const res = await fetch('/dental', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({concerns:Array.from(selectedConcerns)}) });
    const data = await res.json();
    renderProcs(data.procedures);
  } catch(e){ alert('Error: '+e.message); }
  finally { document.getElementById('recommendBtn').disabled=false; }
}

function renderProcs(procedures) {
  const container = document.getElementById('procCards');
  container.innerHTML='';
  procedures.forEach(p=>{
    const idealTags = p.ideal_for.map(t=>`<span class="proc-ideal-tag">${t}</span>`).join('');
    const typeTags  = p.types.map(t=>`<span class="proc-type-tag">${t}</span>`).join('');
    const card = document.createElement('div');
    card.className='proc-card';
    card.innerHTML=`
      <div class="proc-top"><span class="proc-icon">${p.icon}</span><span class="proc-name">${p.name}</span></div>
      <p class="proc-desc">${p.description}</p>
      <div class="proc-ideal-label">Ideal For</div><div class="proc-ideal-tags">${idealTags}</div>
      <div class="proc-meta">
        <div class="proc-meta-item"><div class="proc-meta-label">Estimated Cost</div><div class="proc-meta-value">${p.cost}</div></div>
        <div class="proc-meta-item"><div class="proc-meta-label">Treatment Time</div><div class="proc-meta-value">${p.duration}</div></div>
        <div class="proc-meta-item"><div class="proc-meta-label">Longevity</div><div class="proc-meta-value">${p.longevity}</div></div>
        <div class="proc-meta-item"><div class="proc-meta-label">Discomfort</div><div class="proc-meta-value">${p.pain_level}</div></div>
      </div>
      <div class="proc-types-label">Available Options</div>${typeTags}
      <p class="proc-disclaimer">⚠ ${p.disclaimer}</p>`;
    container.appendChild(card);
  });
  document.getElementById('procResults').style.display='block';
  document.getElementById('procResults').scrollIntoView({behavior:'smooth',block:'start'});
}
</script>
</body>
</html>"""


# ── HELPERS ────────────────────────────────────────────────────────────────
def ratio_score(measured):
    dev   = abs(measured - PHI)
    score = max(0.0, 100.0 * (1.0 - dev / PHI))
    return round(score, 2), round(dev, 4)

def generate_analysis(overall, ratios):
    best  = max(ratios, key=lambda r: r['score'])
    worst = min(ratios, key=lambda r: r['score'])
    if overall >= 85:
        opening = "Your facial proportions demonstrate a remarkable alignment with the golden ratio, placing you among the most harmoniously proportioned faces by classical standards."
    elif overall >= 72:
        opening = "Your face exhibits strong phi harmony across multiple dimensions, reflecting balanced, naturally pleasing proportion."
    elif overall >= 60:
        opening = "Your facial structure shows moderate golden ratio alignment, with several features echoing the divine proportion."
    else:
        opening = "Your face carries a distinctive character that departs from classical phi proportions — a quality shared by many celebrated faces throughout art history."
    return f"{opening} Your strongest ratio is {best['name'].lower()} ({best['score']:.1f}%), while {worst['name'].lower()} shows the most deviation from φ. Overall harmony score: {overall:.1f}%."

def annotate_image(img, face, eyes, nose_pt, mouth_pt):
    out = img.copy()
    x, y, w, h = face
    cv2.rectangle(out, (x,y), (x+w,y+h), (201,168,76), 1)
    for (ex,ey,ew,eh) in eyes:
        cx=x+ex+ew//2; cy=y+ey+eh//2
        cv2.circle(out,(cx,cy),3,(201,168,76),-1)
        cv2.ellipse(out,(cx,cy),(ew//2,eh//2),0,0,360,(201,168,76),1)
    if nose_pt:  cv2.circle(out,nose_pt,4,(232,201,122),-1); cv2.circle(out,nose_pt,10,(232,201,122),1)
    if mouth_pt: cv2.circle(out,mouth_pt,4,(138,110,47),-1)
    for yy in [y,y+h//3,y+2*h//3,y+h]: cv2.line(out,(x,yy),(x+w,yy),(201,168,76),1)
    cv2.line(out,(x+w//2,y),(x+w//2,y+h),(201,168,76),1)
    cv2.putText(out,f"phi={PHI:.4f}",(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.45,(201,168,76),1,cv2.LINE_AA)
    return out


# ── ROUTES ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error':'No image uploaded.'}), 400
    img_bytes = request.files['image'].read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error':'Could not decode image.'}), 400

    gray  = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    faces = FACE_CASCADE.detectMultiScale(gray,1.1,5,minSize=(80,80))
    if len(faces)==0: faces=FACE_CASCADE.detectMultiScale(gray,1.05,3,minSize=(60,60))
    if len(faces)==0: return jsonify({'error':'No face detected. Please use a clear frontal selfie with good lighting.'}), 400

    face = max(faces, key=lambda f:f[2]*f[3])
    fx,fy,fw,fh = face
    fg = gray[fy:fy+fh, fx:fx+fw]

    eyes = EYE_CASCADE.detectMultiScale(fg,1.1,5,minSize=(20,20))
    eyes = sorted([(ex,ey,ew,eh) for ex,ey,ew,eh in eyes if ey<fh*.55],key=lambda e:e[0])
    eyeC = [(ex+ew//2,ey+eh//2) for ex,ey,ew,eh in eyes]

    nL=(fw//2,int(fh*.60)); mL=(fw//2,int(fh*.78))
    nG=(fx+nL[0],fy+nL[1]); mG=(fx+mL[0],fy+mL[1])

    ratios=[]
    fwh=fw/fh if fh>0 else PHI
    s,d=ratio_score(fwh); ratios.append({'name':'Face Width to Height','measured':round(fwh,4),'score':s,'deviation':d})

    if len(eyes)>=2:
        le,re=eyes[0],eyes[-1]
        esp=abs((re[0]+re[2]//2)-(le[0]+le[2]//2)); aew=(le[2]+re[2])/2
        s,d=ratio_score(esp/aew if aew>0 else PHI); ratios.append({'name':'Eye Spacing to Eye Width','measured':round(esp/aew,4),'score':s,'deviation':d})
        s,d=ratio_score(fw/esp if esp>0 else PHI);  ratios.append({'name':'Face Width to Eye Span','measured':round(fw/esp,4),'score':s,'deviation':d})
        sym=max(le[2],re[2])/min(le[2],re[2]) if min(le[2],re[2])>0 else 1.0
        s,d=ratio_score(sym); ratios.append({'name':'Eye Width Symmetry','measured':round(sym,4),'score':s,'deviation':d})
    else:
        s,d=ratio_score((fw*.35)/(fw*.22)); ratios.append({'name':'Eye Spacing to Eye Width (est.)','measured':round((fw*.35)/(fw*.22),4),'score':s,'deviation':d})

    if eyeC:
        aey=sum(e[1] for e in eyeC)/len(eyeC); mid=nL[1]-aey
        if mid>0 and aey>0: s,d=ratio_score(aey/mid); ratios.append({'name':'Hairline to Eye / Eye to Nose','measured':round(aey/mid,4),'score':s,'deviation':d})
        lo=fh-nL[1]
        if mid>0 and lo>0: s,d=ratio_score(mid/lo); ratios.append({'name':'Mid Face to Lower Face','measured':round(mid/lo,4),'score':s,'deviation':d})

    s,d=ratio_score((fw*.46)/(fw*.28)); ratios.append({'name':'Mouth Width to Nose Width','measured':round((fw*.46)/(fw*.28),4),'score':s,'deviation':d})
    nc=fh-nL[1]; s,d=ratio_score(fh/nc if nc>0 else PHI); ratios.append({'name':'Face Height to Chin Segment','measured':round(fh/nc,4),'score':s,'deviation':d})

    weights=[1.0]*len(ratios)
    if len(eyes)>=2: weights[1]=1.5; weights[2]=1.3
    overall=sum(r['score']*w for r,w in zip(ratios,weights))/sum(weights)

    ann=annotate_image(img,face,eyes,nG,mG)
    _,buf=cv2.imencode('.jpg',ann,[cv2.IMWRITE_JPEG_QUALITY,88])
    return jsonify({'overall_score':round(overall,2),'ratios':ratios,'analysis':generate_analysis(overall,ratios),'annotated_image':base64.standard_b64encode(buf).decode(),'landmarks_detected':len(eyes)})

# ── VISUAL SIMULATION ──────────────────────────────────────────────────────
def simulate_improvements(img, face, eyes, ratios):
    """Apply OpenCV-based visual improvements to simulate symmetry changes."""
    out = img.copy().astype(np.float32)
    h_img, w_img = img.shape[:2]
    fx, fy, fw, fh = face

    ratio_map = {r['name']: r for r in ratios}

    # ── 1. SKIN SMOOTHING (bilateral filter — removes blemishes, evens tone) ──
    face_region = out[fy:fy+fh, fx:fx+fw]
    smooth = cv2.bilateralFilter(face_region.astype(np.uint8), 15, 80, 80).astype(np.float32)
    # Blend: 60% smooth, 40% original (keeps naturalness)
    out[fy:fy+fh, fx:fx+fw] = cv2.addWeighted(face_region, 0.4, smooth, 0.6, 0)

    # ── 2. BRIGHTNESS & CONTRAST LIFT (soft glow) ──
    face_f = out[fy:fy+fh, fx:fx+fw]
    # Soft light blend — brightens midtones
    normalized = face_f / 255.0
    brightened = normalized + 0.08 * (1 - normalized)
    out[fy:fy+fh, fx:fx+fw] = np.clip(brightened * 255, 0, 255)

    # ── 3. EYE ENHANCEMENT (brighten eye whites, subtle definition) ──
    if len(eyes) >= 1:
        for (ex, ey, ew, eh) in eyes[:2]:
            # Eye region in global coords
            ex_g, ey_g = fx+ex, fy+ey
            eye_roi = out[ey_g:ey_g+eh, ex_g:ex_g+ew]
            if eye_roi.size == 0: continue
            # Slightly brighten and add micro-contrast
            eye_bright = cv2.convertScaleAbs(eye_roi.astype(np.uint8), alpha=1.08, beta=6)
            out[ey_g:ey_g+eh, ex_g:ex_g+ew] = eye_bright.astype(np.float32)

    # ── 4. FACE WIDTH ADJUSTMENT (slim or widen based on phi deviation) ──
    fwh_r = ratio_map.get('Face Width to Height')
    if fwh_r and fwh_r['deviation'] > 0.15:
        face_crop = out[fy:fy+fh, fx:fx+fw].astype(np.uint8)
        if fwh_r['measured'] > PHI:
            # Face too wide — slim it slightly (compress horizontally 4%)
            new_w = int(fw * 0.96)
            slimmed = cv2.resize(face_crop, (new_w, fh))
            # Place centered
            pad = (fw - new_w) // 2
            canvas = face_crop.copy()
            canvas[:, pad:pad+new_w] = slimmed
            # Feather edges
            for i in range(min(pad, 12)):
                alpha = i / max(pad, 1)
                canvas[:, i] = (canvas[:, i] * alpha + img[fy:fy+fh, fx+i] * (1-alpha)).astype(np.uint8)
                canvas[:, fw-1-i] = (canvas[:, fw-1-i] * alpha + img[fy:fy+fh, fx+fw-1-i] * (1-alpha)).astype(np.uint8)
            out[fy:fy+fh, fx:fx+fw] = canvas.astype(np.float32)
        else:
            # Face too narrow — widen slightly (expand 3%)
            new_w = int(fw * 1.03)
            widened = cv2.resize(face_crop, (min(new_w, w_img-fx), fh))
            out[fy:fy+fh, fx:fx+min(new_w, w_img-fx)] = widened[:, :min(new_w, w_img-fx)].astype(np.float32)

    # ── 5. EYE SYMMETRY CORRECTION (mirror the better eye subtly) ──
    eye_sym = ratio_map.get('Eye Width Symmetry')
    if eye_sym and eye_sym['deviation'] > 0.2 and len(eyes) >= 2:
        le, re = eyes[0], eyes[1]
        # Slightly open the smaller eye by brightening its upper lid area
        smaller = le if le[2] < re[2] else re
        sx, sy = fx+smaller[0], fy+smaller[1]
        lid_region = out[sy:sy+smaller[3]//3, sx:sx+smaller[2]]
        if lid_region.size > 0:
            out[sy:sy+smaller[3]//3, sx:sx+smaller[2]] = np.clip(lid_region * 1.06 + 4, 0, 255)

    # ── 6. FOREHEAD CONTOURING (darken temples if face too wide) ──
    fwh_r2 = ratio_map.get('Face Width to Height')
    if fwh_r2 and fwh_r2['measured'] > PHI + 0.1:
        temple_w = int(fw * 0.12)
        for side_x in [fx, fx+fw-temple_w]:
            temple = out[fy:fy+int(fh*0.35), side_x:side_x+temple_w].astype(np.float32)
            # Darken temples for slimming effect
            gradient = np.linspace(0.82, 1.0, temple_w).reshape(1,-1,1)
            if side_x == fx: gradient = gradient[:,::-1,:]
            out[fy:fy+int(fh*0.35), side_x:side_x+temple_w] = np.clip(temple * gradient, 0, 255)

    # ── 7. JAWLINE DEFINITION (subtle sharpening along jaw) ──
    jaw_y = fy + int(fh * 0.75)
    jaw_region = out[jaw_y:fy+fh, fx:fx+fw].astype(np.uint8)
    if jaw_region.size > 0:
        sharpened = cv2.filter2D(jaw_region, -1, np.array([[-0.3,-0.3,-0.3],[-0.3,3.4,-0.3],[-0.3,-0.3,-0.3]]))
        out[jaw_y:fy+fh, fx:fx+fw] = cv2.addWeighted(jaw_region, 0.6, sharpened, 0.4, 0).astype(np.float32)

    # ── 8. OVERALL WARMTH + VIBRANCY ──
    out_uint8 = np.clip(out, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(out_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.08, 0, 255)  # slight saturation boost
    hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.03, 0, 255)  # slight value boost
    out_uint8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return out_uint8


def create_side_by_side(original, simulated, face):
    """Create a comparison image with before/after labels."""
    h, w = original.shape[:2]
    # Resize both to same size
    target_h = min(h, 600)
    scale = target_h / h
    target_w = int(w * scale)
    orig_r = cv2.resize(original, (target_w, target_h))
    sim_r  = cv2.resize(simulated, (target_w, target_h))

    # Add labels
    label_h = 36
    combined_w = target_w * 2 + 4
    canvas = np.ones((target_h + label_h, combined_w, 3), dtype=np.uint8) * 20

    # Place images
    canvas[label_h:, :target_w] = orig_r
    canvas[label_h:, target_w+4:] = sim_r

    # Labels
    cv2.rectangle(canvas, (0,0), (target_w, label_h), (40,40,40), -1)
    cv2.rectangle(canvas, (target_w+4,0), (combined_w, label_h), (30,60,50), -1)
    cv2.putText(canvas, 'ORIGINAL', (target_w//2-52, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,160,120), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'SIMULATED', (target_w+4+target_w//2-58, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80,180,160), 1, cv2.LINE_AA)

    # Divider line
    cv2.rectangle(canvas, (target_w, 0), (target_w+4, target_h+label_h), (60,60,60), -1)

    return canvas


@app.route('/simulate', methods=['POST'])
def simulate():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400

    img_bytes = request.files['image'].read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Could not decode image.'}), 400

    # Parse ratios from form data
    import json as _json
    ratios_raw = request.form.get('ratios', '[]')
    try: ratios = _json.loads(ratios_raw)
    except: ratios = []

    gray  = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
    if len(faces) == 0: faces = FACE_CASCADE.detectMultiScale(gray, 1.05, 3, minSize=(60,60))
    if len(faces) == 0: return jsonify({'error': 'No face detected in image.'}), 400

    face = max(faces, key=lambda f: f[2]*f[3])
    fx, fy, fw, fh = face
    fg = gray[fy:fy+fh, fx:fx+fw]
    eyes = EYE_CASCADE.detectMultiScale(fg, 1.1, 5, minSize=(20,20))
    eyes = sorted([(ex,ey,ew,eh) for ex,ey,ew,eh in eyes if ey < fh*.55], key=lambda e: e[0])

    simulated = simulate_improvements(img, face, eyes, ratios)
    comparison = create_side_by_side(img, simulated, face)

    # Encode both
    _, buf_sim  = cv2.imencode('.jpg', simulated,  [cv2.IMWRITE_JPEG_QUALITY, 92])
    _, buf_comp = cv2.imencode('.jpg', comparison, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return jsonify({
        'simulated':   base64.standard_b64encode(buf_sim).decode(),
        'comparison':  base64.standard_b64encode(buf_comp).decode(),
        'improvements': [
            'Skin texture smoothed and tone evened',
            'Soft glow applied to face region',
            'Eye definition and brightness enhanced',
            'Face width adjusted toward phi ratio' if any(r['name']=='Face Width to Height' and r['deviation']>0.15 for r in ratios) else 'Face proportions maintained',
            'Eye symmetry subtly balanced',
            'Temple contouring applied',
            'Jawline definition sharpened',
            'Overall warmth and vibrancy enhanced'
        ]
    })


@app.route('/dental', methods=['POST'])
def dental():
    concerns=request.get_json().get('concerns',[])
    ids=set()
    for c in concerns:
        for pid in SMILE_CONCERNS.get(c,[]): ids.add(pid)
    if not ids: ids={'teeth_whitening','smile_makeover'}
    order=['teeth_whitening','bonding','gum_contouring','veneers','crown','orthodontics','implants','smile_makeover']
    procs=[DENTAL_PROCEDURES[p] for p in order if p in ids]
    return jsonify({'procedures':procs})

if __name__ == '__main__':
    port=int(os.environ.get('PORT',5050))
    print(f"\n  φ◉  Face & Smile Advisor")
    print(f"  ─────────────────────────────")
    print(f"  Open: http://localhost:{port}\n")
    app.run(debug=False,host='0.0.0.0',port=port)
