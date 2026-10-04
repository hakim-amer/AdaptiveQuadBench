\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{l4dc2026}[2025/07/31 v1.0 L4DC 2026 Proceedings]

\@ifundefined{ifanonsubmission}{
  \newif\ifanonsubmission
  \anonsubmissionfalse
}{}

\@ifundefined{iffinalsubmission}{
  \newif\iffinalsubmission
  \finalsubmissionfalse
}{}

\DeclareOption{anon}{\anonsubmissiontrue}
\DeclareOption{final}{\finalsubmissiontrue}
\ProcessOptions\relax

\LoadClass[pmlr]{jmlr}

\jmlrvolume{vol vvv}
\jmlryear{2026}
\jmlrproceedings{PMLR}{Proceedings of Machine Learning Research}

\iffinalsubmission
  \editors{G. Sukhatme, L. Lindemann, S. Tu, A. Wierman, N. Atanasov}
  \jmlrworkshop{8th Annual Conference on Learning for Dynamics and Control}
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Temporary workaround fixing issues with \includegraphics (concerns LaTeX2e
%% releases after October 2019) [jsieber; 2020-11-12]
\def\set@curr@file#1{\def\@curr@file{#1}}

%% Workaround for Overleaf [Dean Huang, Will Redman, 2023-10-25]
\makeatletter
   \let\Ginclude@graphics\@org@Ginclude@graphics
\makeatother
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Multiple authors with same address [Idris Kempf 2023-10-30] 
\newcommand{\coltauthor}[1]{\author{#1}}
