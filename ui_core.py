import html
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from QwenRAGSystemOptimized import QwenRAGSystemOptimized

TEAM_COL = "团队名称"
YEAR_COL = "年份"
TRACK_COL = "赛道"
WIKI_COL = "Wiki链接"
PROJ_COL = "项目名称"
SUMMARY_COL = "项目概述"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          :root {
            --bg: #ffd9c8;
            --bg-soft: #ffe8dd;
            --peach: #ffa37f;
            --coral: #ff7b4a;
            --coral-deep: #f26a3d;
            --cream: #fffaf6;
            --panel: rgba(255, 249, 245, 0.92);
            --panel-strong: #fffdfb;
            --line: #f3bea8;
            --line-soft: #f9d8c9;
            --text: #1e1d22;
            --muted: #746763;
            --teal: #7fb5bf;
            --teal-deep: #4a90a4;
            --shadow: 0 18px 44px rgba(162, 92, 53, 0.12);
          }
          .stApp {
            background:
              radial-gradient(560px 560px at -8% 10%, rgba(255, 123, 74, 0.40) 0%, rgba(255, 123, 74, 0.40) 34%, transparent 35%),
              radial-gradient(300px 300px at 88% 12%, rgba(255, 241, 233, 0.95) 0%, rgba(255, 241, 233, 0.95) 30%, transparent 31%),
              radial-gradient(340px 340px at 6% 96%, rgba(255, 166, 135, 0.46) 0%, rgba(255, 166, 135, 0.46) 28%, transparent 29%),
              radial-gradient(420px 420px at 94% 92%, rgba(255, 190, 164, 0.34) 0%, rgba(255, 190, 164, 0.34) 28%, transparent 29%),
              linear-gradient(180deg, #ffd2bd 0%, #ffd8c7 100%);
            color: var(--text);
            font-family: "Trebuchet MS", "Segoe UI", sans-serif;
          }
          [data-testid="stSidebar"] {
            background:
              radial-gradient(220px 120px at 12% 8%, rgba(255, 146, 106, 0.26) 0%, transparent 72%),
              linear-gradient(180deg, #fff4ed 0%, #ffe8db 100%);
            border-right: 1px solid rgba(242, 149, 112, 0.35);
          }
          [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.2rem;
          }
          .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1220px;
          }
          h1, h2, h3, h4 {
            color: var(--text);
            letter-spacing: -0.02em;
          }
          p, label, .stCaption, .stMarkdown, .stText {
            color: var(--text);
          }
          .hero {
            position: relative;
            overflow: hidden;
            background:
              radial-gradient(220px 220px at 12% 22%, rgba(125, 181, 191, 0.22) 0%, transparent 62%),
              radial-gradient(220px 220px at 88% 18%, rgba(255, 186, 160, 0.52) 0%, transparent 58%),
              linear-gradient(180deg, rgba(255, 251, 247, 0.98) 0%, rgba(255, 247, 242, 0.96) 100%);
            color: var(--text);
            border-radius: 34px;
            padding: 34px 34px 36px;
            margin: 10px 0 18px;
            box-shadow: 0 26px 58px rgba(190, 108, 68, 0.16);
            border: 1px solid rgba(244, 186, 160, 0.58);
          }
          .hero:after {
            content: "";
            position: absolute;
            inset: auto 18px 18px auto;
            width: 132px;
            height: 132px;
            border-radius: 999px;
            background:
              radial-gradient(circle at 50% 50%, rgba(126, 181, 191, 0.70) 0%, rgba(88, 153, 170, 0.92) 48%, rgba(57, 115, 130, 1) 100%);
            box-shadow:
              0 0 0 8px rgba(126, 181, 191, 0.16),
              inset 0 0 0 8px rgba(255,255,255,0.12);
            opacity: 0.95;
          }
          .hero:before {
            content: "";
            position: absolute;
            inset: 20px auto auto 20px;
            width: 150px;
            height: 150px;
            border-radius: 999px;
            background:
              radial-gradient(circle at 50% 50%, rgba(135, 205, 223, 0.88) 0%, rgba(92, 154, 176, 0.92) 42%, rgba(41, 88, 103, 0.96) 100%);
            box-shadow:
              0 0 0 10px rgba(127, 181, 191, 0.10),
              inset 0 0 0 10px rgba(255,255,255,0.13);
            opacity: 0.96;
          }
          .hero h1 {
            color: var(--text);
            font-size: 2.45rem;
            margin: 1.25rem 0 0.55rem;
            text-align: center;
            font-weight: 800;
            position: relative;
            z-index: 4;
          }
          .hero p {
            color: var(--muted);
            font-size: 1rem;
            margin-bottom: 0;
            max-width: 760px;
            text-align: center;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.65;
            position: relative;
            z-index: 4;
          }
          .hero-tag {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 10px 24px;
            border-radius: 999px;
            background: linear-gradient(180deg, #ffd1be 0%, #ffc4a8 100%);
            color: var(--text);
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            box-shadow: 0 10px 24px rgba(239, 139, 95, 0.18);
            margin: 0 auto;
            position: relative;
            z-index: 4;
            width: fit-content;
          }
          .hero-ornament {
            position: absolute;
            z-index: 2;
            pointer-events: none;
            opacity: 0.78;
          }
          .hero-ornament.blob-left {
            left: 164px;
            top: 28px;
            width: 66px;
            height: 66px;
            border-radius: 999px;
            background:
              radial-gradient(circle at 36% 34%, rgba(255,255,255,0.62) 0 8px, transparent 9px),
              radial-gradient(circle at 64% 44%, rgba(175, 46, 30, 0.82) 0 8px, transparent 9px),
              radial-gradient(circle at 44% 68%, rgba(175, 46, 30, 0.72) 0 6px, transparent 7px),
              linear-gradient(180deg, rgba(123, 196, 210, 0.92) 0%, rgba(63, 127, 143, 0.96) 100%);
            box-shadow:
              0 0 0 8px rgba(124, 187, 201, 0.10),
              inset 0 0 0 6px rgba(255,255,255,0.10);
          }
          .hero-ornament.blob-left:after {
            content: "";
            position: absolute;
            right: -14px;
            top: 24px;
            width: 28px;
            height: 10px;
            border-radius: 999px;
            background:
              radial-gradient(circle at 10% 50%, rgba(242, 168, 66, 0.95) 0 4px, transparent 5px),
              radial-gradient(circle at 35% 50%, rgba(242, 168, 66, 0.95) 0 4px, transparent 5px),
              radial-gradient(circle at 60% 50%, rgba(242, 168, 66, 0.95) 0 4px, transparent 5px),
              radial-gradient(circle at 85% 50%, rgba(242, 168, 66, 0.95) 0 4px, transparent 5px);
          }
          .hero-ornament.blob-mid {
            left: 50%;
            top: 34px;
            width: 120px;
            height: 30px;
            margin-left: -62px;
            background: transparent;
            box-shadow: none;
          }
          .hero-ornament.blob-mid:before {
            content: "";
            position: absolute;
            inset: 11px 10px auto 10px;
            height: 8px;
            border-radius: 999px;
            background:
              radial-gradient(circle at 0% 50%, rgba(81, 153, 162, 0.94) 0 6px, transparent 7px),
              radial-gradient(circle at 20% 50%, rgba(81, 153, 162, 0.94) 0 6px, transparent 7px),
              radial-gradient(circle at 40% 50%, rgba(81, 153, 162, 0.94) 0 6px, transparent 7px),
              radial-gradient(circle at 60% 50%, rgba(81, 153, 162, 0.94) 0 6px, transparent 7px),
              radial-gradient(circle at 80% 50%, rgba(81, 153, 162, 0.94) 0 6px, transparent 7px),
              radial-gradient(circle at 100% 50%, rgba(81, 153, 162, 0.94) 0 6px, transparent 7px);
          }
          .hero-ornament.blob-mid:after {
            content: "";
            position: absolute;
            left: -14px;
            top: 6px;
            width: 18px;
            height: 18px;
            border-radius: 999px;
            background: rgba(150, 207, 198, 0.92);
            box-shadow: 128px 0 0 rgba(233, 161, 144, 0.92);
          }
          .hero-ornament.blob-right {
            right: 156px;
            top: 34px;
            width: 62px;
            height: 62px;
            border-radius: 999px;
            background:
              radial-gradient(circle at 34% 32%, rgba(255,255,255,0.66) 0 8px, transparent 9px),
              radial-gradient(circle at 66% 38%, rgba(255, 125, 96, 0.76) 0 7px, transparent 8px),
              radial-gradient(circle at 42% 66%, rgba(255, 125, 96, 0.64) 0 5px, transparent 6px),
              linear-gradient(180deg, rgba(227, 238, 255, 0.96) 0%, rgba(165, 193, 246, 0.92) 100%);
            box-shadow:
              0 0 0 8px rgba(184, 209, 250, 0.12),
              inset 0 0 0 5px rgba(255,255,255,0.16);
          }
          .hero-ornament.blob-right:before {
            content: "";
            position: absolute;
            left: -38px;
            top: 27px;
            width: 30px;
            height: 8px;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(87, 153, 163, 0.92) 0%, rgba(87, 153, 163, 0.30) 100%);
          }
          .hero-ornament.blob-right:after {
            content: "";
            position: absolute;
            left: -50px;
            top: 19px;
            width: 14px;
            height: 14px;
            border-radius: 999px;
            background: rgba(87, 153, 163, 0.92);
          }
          .hero-ai-face {
            position: absolute;
            right: 28px;
            top: 18px;
            width: 96px;
            height: 96px;
            border-radius: 999px;
            background:
              radial-gradient(circle at 32% 32%, rgba(255,255,255,0.7) 0 8px, transparent 9px),
              radial-gradient(circle at 72% 28%, rgba(255,102,92,0.9) 0 11px, transparent 12px),
              linear-gradient(180deg, #d8e8ff 0%, #a7c7fb 100%);
            border: 4px solid rgba(32, 33, 38, 0.9);
            box-shadow: 0 12px 24px rgba(116, 103, 99, 0.12);
            z-index: 2;
            opacity: 0.86;
          }
          .hero-ai-face:before {
            content: "";
            position: absolute;
            left: 24px;
            top: 38px;
            width: 12px;
            height: 18px;
            border-radius: 999px;
            background: #2652c6;
            box-shadow: 34px 0 0 #2652c6;
          }
          .hero-ai-face:after {
            content: "";
            position: absolute;
            left: 35px;
            top: 60px;
            width: 32px;
            height: 16px;
            border-bottom: 4px solid #2652c6;
            border-radius: 0 0 22px 22px;
          }
          .hero-ai-whisker {
            position: absolute;
            width: 18px;
            height: 3px;
            background: rgba(32, 33, 38, 0.9);
            border-radius: 999px;
            z-index: 2;
          }
          .hero-ai-whisker.w1 { right: 8px; top: 46px; transform: rotate(12deg); }
          .hero-ai-whisker.w2 { right: 8px; top: 64px; transform: rotate(-8deg); }
          .hero-ai-whisker.w3 { right: 102px; top: 46px; transform: rotate(-12deg); }
          .hero-ai-whisker.w4 { right: 102px; top: 64px; transform: rotate(8deg); }
          .hero-ai-whisker.w5 { right: 38px; top: 12px; transform: rotate(74deg); }
          .hero-ai-whisker.w6 { right: 56px; top: 10px; transform: rotate(104deg); }
          .metric-card,.insight-card,.context-box,.result-box,.plan-card,.soft-panel,.section-card {
            background: var(--panel);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(244, 186, 160, 0.54);
            border-radius: 24px;
            padding: 14px 16px;
            box-shadow: var(--shadow);
          }
          .metric-card {
            min-height: 92px;
            position: relative;
            overflow: hidden;
          }
          .metric-card:after {
            content: "";
            position: absolute;
            inset: auto -18px -22px auto;
            width: 76px;
            height: 76px;
            border-radius: 999px;
            background: rgba(127, 181, 191, 0.14);
          }
          .metric-card div:first-child {
            color: var(--muted);
            font-size: 0.92rem;
            margin-bottom: 0.3rem;
          }
          .metric-card b {
            font-size: 1.5rem;
            color: var(--text);
          }
          .insight-card {
            min-height: 118px;
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
            position: relative;
            overflow: hidden;
          }
          .insight-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 22px 42px rgba(181, 105, 68, 0.18);
            border-color: #f0b89d;
          }
          .insight-card:after {
            content: "";
            position: absolute;
            right: -16px;
            top: -18px;
            width: 78px;
            height: 78px;
            border-radius: 999px;
            background: rgba(127, 181, 191, 0.14);
          }
          .insight-card div:first-child {
            font-size: 1.03rem;
            margin-bottom: 0.35rem;
          }
          .insight-card div:last-child {
            color: var(--muted);
            line-height: 1.55;
          }
          .page-shell {
            display: flex;
            flex-direction: column;
            gap: 0.9rem;
          }
          .section-title {
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            color: var(--coral-deep);
            margin-bottom: 0.45rem;
            font-weight: 800;
          }
          .section-heading {
            font-size: 1.38rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.25rem;
          }
          .section-desc {
            color: var(--muted);
            margin-bottom: 0;
          }
          .page-nav {
            display:flex;
            gap:10px;
            margin: 0 0 14px;
            flex-wrap: wrap;
            padding: 6px 2px 2px;
          }
          .page-chip {
            display:inline-block;
            padding:10px 15px;
            border-radius:999px;
            border:1px solid rgba(244, 186, 160, 0.58);
            background: rgba(255, 248, 243, 0.82);
            color: var(--text);
            text-decoration:none;
            font-size:13px;
            font-weight: 600;
            box-shadow: 0 8px 18px rgba(186, 114, 76, 0.08);
            transition: all .18s ease;
          }
          .page-chip:hover {
            transform: translateY(-1px);
            border-color: #edae92;
          }
          .page-chip.active {
            background: linear-gradient(135deg, #ff976d 0%, #ff7b4a 100%);
            color:#fffdfb;
            border-color: transparent;
            box-shadow: 0 16px 30px rgba(242, 106, 61, 0.24);
          }
          .context-box {
            margin-bottom: 0.7rem;
            line-height: 1.6;
          }
          .result-box {
            line-height: 1.72;
            font-size: 0.98rem;
          }
          .plan-card {
            min-height: 70px;
          }
          .soft-panel {
            margin: 0.45rem 0 1rem;
          }
          .toolbar-note {
            color: var(--muted);
            font-size: 0.92rem;
            margin: 0 0 0.5rem;
          }
          .user-bubble {
            background: linear-gradient(135deg, #ff976d 0%, #ff7b4a 100%);
            color:#fff;
            border-radius:24px 24px 8px 24px;
            padding:12px 14px;
            max-width: 90%;
            margin-left:auto;
            box-shadow: 0 16px 32px rgba(242, 106, 61, 0.18);
            border: 1px solid rgba(255,255,255,0.1);
          }
          .assistant-bubble {
            background: rgba(255, 252, 249, 0.96);
            color: var(--text);
            border:1px solid rgba(244, 186, 160, 0.50);
            border-radius:24px 24px 24px 8px;
            padding:12px 14px;
            max-width:92%;
            box-shadow: 0 14px 30px rgba(186, 114, 76, 0.10);
          }
          div[data-testid="stForm"] {
            background: rgba(255, 250, 246, 0.84);
            border: 1px solid rgba(244, 186, 160, 0.56);
            border-radius: 24px;
            padding: 12px 14px 2px;
            box-shadow: var(--shadow);
          }
          div[data-testid="stDataFrame"] {
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid rgba(244, 186, 160, 0.52);
            box-shadow: 0 12px 28px rgba(186, 114, 76, 0.10);
          }
          .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
            border-radius: 16px;
            border: 1px solid rgba(242, 149, 112, 0.20);
            background: linear-gradient(180deg, #fffefe 0%, #fff5ef 100%);
            color: var(--text);
            min-height: 2.8rem;
            font-weight: 600;
            box-shadow: 0 10px 22px rgba(186, 114, 76, 0.08);
            transition: all .18s ease;
          }
          .stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
            border-color: rgba(242, 149, 112, 0.40);
            transform: translateY(-1px);
          }
          .stChatInputContainer {
            background: rgba(255, 250, 246, 0.88);
            border: 1px solid rgba(244, 186, 160, 0.52);
            border-radius: 22px;
            box-shadow: var(--shadow);
          }
          .stTextInput input, .stTextArea textarea {
            border-radius: 16px !important;
          }
          .stMultiSelect div[data-baseweb="select"], .stSelectbox div[data-baseweb="select"] {
            border-radius: 16px;
          }
          .stExpander {
            border: 1px solid rgba(244, 186, 160, 0.54);
            border-radius: 20px;
            background: rgba(255, 250, 246, 0.82);
            box-shadow: 0 10px 24px rgba(186, 114, 76, 0.08);
          }
          .bubble-field {
            position: fixed;
            inset: 0;
            pointer-events: none;
            overflow: hidden;
            z-index: 0;
          }
          .bubble {
            position: absolute;
            bottom: -120px;
            border-radius: 999px;
            background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.96) 0%, rgba(255,255,255,0.54) 28%, rgba(138, 202, 214, 0.22) 100%);
            border: 1px solid rgba(255,255,255,0.35);
            opacity: 0;
            animation: floatUp linear infinite;
            box-shadow: 0 0 0 1px rgba(126, 181, 191, 0.08);
          }
          .bubble.b1 { left: 2.5%; width: 26px; height: 26px; animation-duration: 10s; animation-delay: 0s; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.96) 0%, rgba(255,255,255,0.56) 28%, rgba(184, 223, 233, 0.32) 100%); }
          .bubble.b2 { left: 4.8%; width: 16px; height: 16px; animation-duration: 12.5s; animation-delay: 1.4s; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.96) 0%, rgba(255,255,255,0.54) 28%, rgba(255, 220, 205, 0.34) 100%); }
          .bubble.b3 { left: 7.2%; width: 34px; height: 34px; animation-duration: 11.8s; animation-delay: 3.7s; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.96) 0%, rgba(255,255,255,0.58) 28%, rgba(207, 231, 209, 0.34) 100%); }
          .bubble.b4 { left: 10.8%; width: 22px; height: 22px; animation-duration: 14.5s; animation-delay: 0.9s; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.96) 0%, rgba(255,255,255,0.54) 28%, rgba(212, 222, 252, 0.32) 100%); }
          .bubble.b5 { left: 13.2%; width: 14px; height: 14px; animation-duration: 10.8s; animation-delay: 2.6s; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.96) 0%, rgba(255,255,255,0.54) 28%, rgba(255, 228, 188, 0.34) 100%); }
          .bubble.b6 { left: 85%; width: 30px; height: 30px; animation-duration: 12.2s; animation-delay: 4.8s; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.96) 0%, rgba(255,255,255,0.58) 28%, rgba(184, 223, 233, 0.32) 100%); }
          .bubble.b7 { left: 88%; width: 18px; height: 18px; animation-duration: 13.6s; animation-delay: 1.2s; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.96) 0%, rgba(255,255,255,0.54) 28%, rgba(255, 220, 205, 0.34) 100%); }
          .bubble.b8 { left: 91%; width: 34px; height: 34px; animation-duration: 12.8s; animation-delay: 3.5s; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.96) 0%, rgba(255,255,255,0.58) 28%, rgba(207, 231, 209, 0.34) 100%); }
          .bubble.b9 { left: 94%; width: 16px; height: 16px; animation-duration: 11.4s; animation-delay: 2.2s; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.96) 0%, rgba(255,255,255,0.54) 28%, rgba(212, 222, 252, 0.32) 100%); }
          .bubble.b10 { left: 96.2%; width: 26px; height: 26px; animation-duration: 15s; animation-delay: 4.3s; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.96) 0%, rgba(255,255,255,0.54) 28%, rgba(255, 228, 188, 0.34) 100%); }
          .bubble.b11 { left: 6%; width: 42px; height: 42px; animation-duration: 16s; animation-delay: 5.6s; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.96) 0%, rgba(255,255,255,0.60) 28%, rgba(198, 233, 240, 0.36) 100%); }
          .bubble.b12 { left: 90.5%; width: 42px; height: 42px; animation-duration: 16.4s; animation-delay: 6.2s; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.96) 0%, rgba(255,255,255,0.60) 28%, rgba(255, 213, 220, 0.36) 100%); }
          .bubble.b13 { left: 1.5%; width: 12px; height: 12px; animation-duration: 9.8s; animation-delay: 2.1s; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.96) 0%, rgba(255,255,255,0.54) 28%, rgba(210, 225, 255, 0.34) 100%); }
          .bubble.b14 { left: 98%; width: 12px; height: 12px; animation-duration: 9.8s; animation-delay: 2.9s; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.96) 0%, rgba(255,255,255,0.54) 28%, rgba(255, 228, 188, 0.34) 100%); }
          @keyframes floatUp {
            0% {
              transform: translateY(0) scale(0.92);
              opacity: 0;
            }
            12% {
              opacity: 0.76;
            }
            64% {
              opacity: 0.52;
            }
            100% {
              transform: translateY(-78vh) scale(1.22);
              opacity: 0;
            }
          }
          @media (max-width: 900px) {
            .hero {
              padding: 22px 18px 24px;
              border-radius: 28px;
            }
            .hero h1 {
              font-size: 1.85rem;
            }
            .hero:before,
            .hero:after {
              opacity: 0.22;
              transform: scale(0.82);
            }
            .hero-ornament {
              opacity: 0.18;
              transform: scale(0.75);
            }
            .hero-ai-face,
            .hero-ai-whisker {
              opacity: 0.22;
              transform: scale(0.8);
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_nav(active: str) -> None:
    pages = [
        ("首页", "/"),
        ("知识检索", "/知识检索台"),
        ("推理探索", "/推理探索"),
    ]
    chips = []
    for name, url in pages:
        cls = "page-chip active" if name == active else "page-chip"
        chips.append(f'<a class="{cls}" href="{url}" target="_self">{name}</a>')
    st.markdown(f"<div class='page-nav'>{''.join(chips)}</div>", unsafe_allow_html=True)


def render_section_intro(title: str, desc: str, eyebrow: str = "Workspace") -> None:
    st.markdown(
        (
            "<div class='soft-panel'>"
            f"<div class='section-title'>{eyebrow}</div>"
            f"<div class='section-heading'>{title}</div>"
            f"<p class='section-desc'>{desc}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_hero(title: str, desc: str, tag: str) -> None:
    st.markdown(
        (
            "<div class='hero'>"
            "<div class='hero-ornament blob-left'></div>"
            "<div class='hero-ornament blob-mid'></div>"
            "<div class='hero-ornament blob-right'></div>"
            "<div class='hero-ai-face'></div>"
            "<div class='hero-ai-whisker w1'></div>"
            "<div class='hero-ai-whisker w2'></div>"
            "<div class='hero-ai-whisker w3'></div>"
            "<div class='hero-ai-whisker w4'></div>"
            "<div class='hero-ai-whisker w5'></div>"
            "<div class='hero-ai-whisker w6'></div>"
            f"<div class='hero-tag'>{tag}</div>"
            f"<h1>{title}</h1>"
            f"<p>{desc}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_background_decor() -> None:
    st.markdown(
        """
        <div class="bubble-field" aria-hidden="true">
          <span class="bubble b1"></span>
          <span class="bubble b2"></span>
          <span class="bubble b3"></span>
          <span class="bubble b4"></span>
          <span class="bubble b5"></span>
          <span class="bubble b6"></span>
          <span class="bubble b7"></span>
          <span class="bubble b8"></span>
          <span class="bubble b9"></span>
          <span class="bubble b10"></span>
          <span class="bubble b11"></span>
          <span class="bubble b12"></span>
          <span class="bubble b13"></span>
          <span class="bubble b14"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def detect_user_lang(text: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", text or "") else "en"


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    s = str(text).replace("\r\n", "\n")
    # Merge broken single-character lines like "传\n感\n器"
    pattern = r"((?:[A-Za-z0-9\u4e00-\u9fff]\s*\n\s*){2,}[A-Za-z0-9\u4e00-\u9fff])"
    s = re.sub(pattern, lambda m: re.sub(r"\s*\n\s*", "", m.group(1)), s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def normalize_list(items) -> List[str]:
    if items is None:
        return []
    if isinstance(items, str):
        items = [items]
    out = []
    for x in items:
        t = normalize_text(str(x))
        if t:
            out.append(t)
    return out


def parse_multi_with_unlimited(selected: List, all_options: List) -> List:
    if (not selected) or ("不限" in selected):
        return []
    return [x for x in selected if x in all_options]


@st.cache_resource(show_spinner=False)
def build_system_cached(api_key: str):
    return QwenRAGSystemOptimized(api_key=api_key or None)


def get_rag_system(api_key: str):
    return build_system_cached(api_key or "")


def sidebar_global_api_key() -> str:
    st.sidebar.markdown("## iGEM 助手")
    st.sidebar.caption("竞赛检索与洞察")

    if "global_api_key_saved" not in st.session_state:
        st.session_state.global_api_key_saved = ""
    if "global_api_key_input" not in st.session_state:
        st.session_state.global_api_key_input = ""

    val = st.sidebar.text_input(
        "DASHSCOPE_API_KEY（全局，只需填一次）",
        type="password",
        key="global_api_key_input",
        placeholder="sk-xxxxxxxx",
    )
    if val.strip():
        st.session_state.global_api_key_saved = val.strip()

    source = "会话缓存" if st.session_state.global_api_key_saved else "环境变量"
    st.sidebar.caption(f"当前使用: {source}")
    if st.session_state.global_api_key_saved and st.sidebar.button("清除已保存 Key", use_container_width=True):
        st.session_state.global_api_key_saved = ""
        st.session_state.global_api_key_input = ""
        st.rerun()

    return st.session_state.global_api_key_saved


@st.cache_data(show_spinner=False)
def load_teams_table() -> pd.DataFrame:
    try:
        df = pd.read_csv("out_embedding/igem_teams.csv", encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv("out_embedding/igem_teams.csv", encoding="utf-8")

    required_cols = [
        "Team ID",
        TEAM_COL,
        YEAR_COL,
        TRACK_COL,
        WIKI_COL,
        PROJ_COL,
        SUMMARY_COL,
        "medals",
        "awards_special_prizes",
        "nominations",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    df = df.fillna("").reset_index(drop=True)
    df["row_id"] = df.index
    df["year_num"] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df["awards_text"] = (
        df["medals"].astype(str)
        + " | "
        + df["awards_special_prizes"].astype(str)
        + " | "
        + df["nominations"].astype(str)
    )
    df["search_text"] = (
        df[TEAM_COL].astype(str)
        + " "
        + df[TRACK_COL].astype(str)
        + " "
        + df[PROJ_COL].astype(str)
        + " "
        + df[SUMMARY_COL].astype(str)
        + " "
        + df["awards_text"].astype(str)
    )
    return df


@st.cache_resource(show_spinner=False)
def build_similarity_index(search_texts: Tuple[str, ...]):
    vec = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2)
    return vec, vec.fit_transform(search_texts)


def get_similarity_artifacts(df: pd.DataFrame):
    # Avoid re-hashing a large text tuple on every Streamlit rerun.
    cache_key = (
        int(len(df)),
        int(df["row_id"].max()) if len(df) else 0,
    )
    if st.session_state.get("similarity_cache_key") != cache_key:
        texts = tuple(df["search_text"].astype(str).tolist())
        vec, mat = build_similarity_index(texts)
        st.session_state["similarity_cache_key"] = cache_key
        st.session_state["similarity_vec"] = vec
        st.session_state["similarity_mat"] = mat
    return st.session_state["similarity_vec"], st.session_state["similarity_mat"]


def award_keyword_options(df: pd.DataFrame, topn: int = 80) -> List[str]:
    c = Counter()
    for txt in df["awards_text"].tolist():
        for tok in re.split(r"[;,/|、，\n]+", str(txt)):
            tok = tok.strip()
            if 1 < len(tok) < 48:
                c[tok] += 1
    return [k for k, _ in c.most_common(topn)]


def filter_projects(df: pd.DataFrame, keyword: str, years: List[int], tracks: List[str], awards: List[str]) -> pd.DataFrame:
    out = df.copy()
    if keyword.strip():
        patt = re.escape(keyword.strip())
        m = (
            out[TEAM_COL].str.contains(patt, case=False, regex=True)
            | out[PROJ_COL].str.contains(patt, case=False, regex=True)
            | out[SUMMARY_COL].str.contains(patt, case=False, regex=True)
            | out[TRACK_COL].str.contains(patt, case=False, regex=True)
        )
        out = out[m]
    if years:
        out = out[out["year_num"].isin(years)]
    if tracks:
        out = out[out[TRACK_COL].isin(tracks)]
    if awards:
        patt = "|".join(re.escape(k) for k in awards)
        out = out[out["awards_text"].str.contains(patt, case=False, regex=True)]
    return out.sort_values(by=["year_num", TEAM_COL], ascending=[False, True]).reset_index(drop=True)


def recommend_similar(df: pd.DataFrame, matrix, row_idx: int, topn: int = 5) -> pd.DataFrame:
    sims = cosine_similarity(matrix[row_idx], matrix).flatten()
    ranked = sims.argsort()[::-1]
    rows = []
    for idx in ranked:
        if idx == row_idx:
            continue
        rows.append(
            {
                "相似度": round(float(sims[idx]), 3),
                "年份": int(df.iloc[idx]["year_num"]) if pd.notna(df.iloc[idx]["year_num"]) else "",
                "团队": df.iloc[idx][TEAM_COL],
                "赛道": df.iloc[idx][TRACK_COL],
                "项目": df.iloc[idx][PROJ_COL],
            }
        )
        if len(rows) >= topn:
            break
    return pd.DataFrame(rows)


def run_query_result(rag, question: str) -> Dict:
    with st.spinner("正在检索并生成分析..."):
        return rag.query(question)


def render_sources(sources: List[Dict], title: str = "证据来源"):
    with st.expander(title, expanded=False):
        for s in sources:
            st.write(f"- {s.get('team_name', '未知队伍')}（{s.get('year', '未知年份')}）")


def render_result_block(result: Dict, title: Optional[str] = None):
    if title:
        st.markdown(f"**{title}**")
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.markdown(normalize_text(result.get("answer", "")))
    st.markdown("</div>", unsafe_allow_html=True)
    render_sources(result.get("sources", []))


def render_chat_bubble(role: str, text: str):
    safe = html.escape(normalize_text(str(text))).replace("\n", "<br>")
    c1, c2 = st.columns([1, 1])
    if role == "user":
        with c2:
            st.markdown(f"<div class='user-bubble'>{safe}</div>", unsafe_allow_html=True)
    else:
        with c1:
            st.markdown(f"<div class='assistant-bubble'>{safe}</div>", unsafe_allow_html=True)
