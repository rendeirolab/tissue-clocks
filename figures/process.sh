#!/usr/bin/env bash
# 
# Convert and assemble figures for paper submission.
# 
# Dependencies:
# 1. GNU/Linux
# 2. inkscape (local version preferred, snap or flatpak versions also possible)
# 3. pdfunite
# 4. (optional) minify (https://github.com/tdewolff/minify/tree/master/cmd/minify)
#    `sudo apt install minify`

# Assumptions:
# 1. figures are made in SVG format
# 2. figure files are inside a "svg" directory under ${ROOT_DIR}
# 3. figures are consecutively and consistently labeled
# 3a. Main figures are named Figure<X>.svg and supplementary as FigureS<X>.svg,
#     where <X> is a number (multiple digits allowed).
# 4. (optional) Main figures have a "Figure <X>" and Supplementary Figures a
#    "Supplementary Figure <X>" SVG text label where <X> is a number (multiple digits allowed).

# Editing tips:
# 1. Remove clones and most non-essential grouping from the SVG
# 2. Rasterize large elements such as heatmaps, swarmplots, etc and export them as 300dpi png
# 3. Sometimes it's better not to rasterize but immediately minify an SVG.

echo "Preparing manuscript figures"

ROOT_DIR=$(pwd)
readarray -t MAIN_FIGURES < <(find svg -maxdepth 1 -regextype posix-extended -regex '.*Figure[[:digit:]]+\.svg' | sort)
readarray -t SUPP_FIGURES < <(find svg -maxdepth 1 -regextype posix-extended -regex '.*FigureS[[:digit:]]+\.svg' | sort)
FIGURES=("${MAIN_FIGURES[@]}" "${SUPP_FIGURES[@]}")
NUMBER_MAIN_FIGURES=${#MAIN_FIGURES[@]}
NUMBER_SUPP_FIGURES=${#SUPP_FIGURES[@]}
CURRENT_DATE=$(date '+%Y%m%d')
MINIFY="FALSE"
CLEANUP_TEMP="TRUE"
DPI=300
# INKSCAPE="flatpak run org.inkscape.Inkscape"  # not good, use native or flatpak-spawn (older)
INKSCAPE="inkscape"
MAX_TASKS=16  # number of concurrent inkscape processes

echo "Working in '$ROOT_DIR' directory."

echo ""
echo -e "Found ${NUMBER_MAIN_FIGURES} main figures: \n ${MAIN_FIGURES[@]}"
echo -e "Found ${NUMBER_SUPP_FIGURES} supplementary figures: \n ${SUPP_FIGURES[@]}"

cd "$ROOT_DIR"
mkdir -p {svg/_minified,pdf,png}

if [ "$MINIFY" == "TRUE" ]; then
    echo "Minifying SVG figures."
    SOURCE_DIR=svg/_minified
    for FIGURE in "${FIGURES[@]}"; do
        echo "Figure: $FIGURE"
        minify --type svg --svg-precision 3 --output "${FIGURE/svg/svg\/_minified}" "$FIGURE"
    done
else
    SOURCE_DIR=svg
fi

# Simple function to manage parallel background jobs
wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_TASKS" ]; do
        sleep 0.5
    done
}

echo ""
echo "Exporting figures into PDF (parallelized, max $MAX_TASKS jobs)"
for FIGURE in "${FIGURES[@]}"; do
    wait_for_slot
    {
        echo "Figure: $FIGURE"
        $INKSCAPE \
            --export-type=pdf \
            --export-dpi="$DPI" \
            -o ${FIGURE//svg/pdf} \
            "$FIGURE" # 2> /dev/null
    } &
done
wait

pdfunite "${MAIN_FIGURES[@]//svg/pdf}" MainFigures."${CURRENT_DATE}".pdf
pdfunite "${SUPP_FIGURES[@]//svg/pdf}" SupplementaryFigures."${CURRENT_DATE}".pdf
pdfunite MainFigures."${CURRENT_DATE}".pdf SupplementaryFigures."${CURRENT_DATE}".pdf AllFigures."${CURRENT_DATE}".pdf

echo ""
echo "Producing trimmed, unlabeled figures (parallelized, max $MAX_TASKS jobs)"
for FIGURE in "${FIGURES[@]}"; do
    wait_for_slot
    {
        echo "Figure: $FIGURE"
        NUM=$(echo "$FIGURE" | sed -n "s/^.*FigureS\{0,1\}\(.*\).svg$/\1/p")
        if [[ "$FIGURE" == *"FigureS"* ]]; then
            sed "s/Supplementary Figure $NUM//g" "$FIGURE" > "${FIGURE/.svg/.trimmed.svg}"
        else
            sed "s/Figure $NUM//g" "$FIGURE" > "${FIGURE/.svg/.trimmed.svg}"
        fi

        OUTPUT=${FIGURE/.svg/.trimmed.pdf}
        $INKSCAPE \
            --export-area-drawing \
            --export-margin=5 \
            --export-dpi="$DPI" \
            --export-type=pdf \
            -o "${OUTPUT//svg/trimmed.pdf}" \
            "${FIGURE/.svg/.trimmed.svg}" # 2> /dev/null

        OUTPUT=${FIGURE/.svg/.trimmed.png}
        $INKSCAPE \
            --export-area-drawing \
            --export-margin=5 \
            --export-background=white \
            --export-dpi="$DPI" \
            --export-type=png \
            -o "${OUTPUT/svg/png}" \
            "${FIGURE/.svg/.trimmed.svg}" # 2> /dev/null
    } &
done
wait

PDFS=${MAIN_FIGURES[@]//svg/pdf}
pdfunite ${PDFS[@]//pdf/trimmed.pdf} MainFigures."${CURRENT_DATE}".trimmed.pdf
PDFS=${SUPP_FIGURES[@]//svg/pdf}
pdfunite ${PDFS[@]//pdf/trimmed.pdf} SupplementaryFigures."${CURRENT_DATE}".trimmed.pdf
pdfunite MainFigures."${CURRENT_DATE}".trimmed.pdf SupplementaryFigures."${CURRENT_DATE}".trimmed.pdf AllFigures."${CURRENT_DATE}".trimmed.pdf

if [ "$CLEANUP_TEMP" == "TRUE" ]; then
    rm "${FIGURES[@]/.svg/.trimmed.svg}"
    rm -r trimmed.pdf
fi

echo "Done."
