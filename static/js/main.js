// ToxGDL Client Controller - 3D Visualizer and API Interface
document.addEventListener("DOMContentLoaded", () => {
    // UI Elements
    const smilesInput = document.getElementById("smilesInput");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const presetBadges = document.querySelectorAll(".preset-badge");
    const loadingOverlay = document.getElementById("loadingOverlay");
    
    // Properties panel
    const propsPanel = document.getElementById("propsPanel");
    const valFormula = document.getElementById("valFormula");
    const valMW = document.getElementById("valMW");
    const valLogP = document.getElementById("valLogP");
    const valTPSA = document.getElementById("valTPSA");
    const valHBD = document.getElementById("valHBD");
    const valHBA = document.getElementById("valHBA");
    
    // 3D Viewer Elements
    const viewerElement = document.getElementById("viewer3d");
    const emptyStateText = document.getElementById("emptyStateText");
    const attrSelect = document.getElementById("attrSelect");
    
    // Scores and Progress Bars
    const scores = {
        herg: { text: document.getElementById("scoreHerg"), fill: document.getElementById("fillHerg") },
        ames: { text: document.getElementById("scoreAmes"), fill: document.getElementById("fillAmes") },
        dili: { text: document.getElementById("scoreDili"), fill: document.getElementById("fillDili") },
        ld50: { text: document.getElementById("scoreLd50"), fill: document.getElementById("fillLd50") }
    };
    
    // Copilot Panel
    const copilotPanel = document.getElementById("copilotPanel");
    const copilotBody = document.getElementById("copilotBody");
    
    // Active prediction data store
    let currentMoleculeData = null;
    let glViewer = null;

    // Helper: Map Integrated Gradients score [-1.0, 1.0] to a stunning RGB color
    function getAttributionColor(score) {
        // Safe: Green, Toxic: Red, Neutral: White/Light Grey
        if (score > 0) {
            // Map positive score to gradient: White [255, 255, 255] -> Deep Red [239, 68, 68]
            const r = 255;
            const g = Math.round(255 - score * 187);
            const b = Math.round(255 - score * 187);
            return `rgb(${r}, ${g}, ${b})`;
        } else if (score < 0) {
            // Map negative score to gradient: White [255, 255, 255] -> Vibrant Green [16, 185, 129]
            const absScore = Math.abs(score);
            const r = Math.round(255 - absScore * 239);
            const g = 255;
            const b = Math.round(255 - absScore * 126);
            return `rgb(${r}, ${g}, ${b})`;
        } else {
            return "rgb(240, 240, 240)";
        }
    }

    // Helper: Determine Progress Bar Color Theme based on prediction probability
    function getThemeClass(prob) {
        if (prob < 0.3) return "safe";
        if (prob <= 0.6) return "warning";
        return "toxic";
    }

    // Initialize/Redraw 3D Molecular Conformation
    function render3DConformer() {
        if (!currentMoleculeData || !currentMoleculeData.sdf) {
            emptyStateText.style.display = "block";
            return;
        }
        
        emptyStateText.style.display = "none";
        
        // Reset container content for 3Dmol
        viewerElement.innerHTML = "";
        
        if (typeof $3Dmol === "undefined") {
            viewerElement.innerHTML = `<div class="empty-state-text" style="color: var(--toxic); font-weight: 600; padding: 20px; line-height: 1.6;">
                ⚠️ 3Dmol.js Library Error:<br>
                Unable to render 3D stereochemical conformer because the 3Dmol library failed to load in your browser.<br><br>
                Please verify your internet connection or check your browser console for script blocker extensions.
            </div>`;
            return;
        }
        
        // Initialize 3Dmol viewer
        glViewer = $3Dmol.createViewer(viewerElement, { backgroundColor: "#060813" });
        glViewer.addModel(currentMoleculeData.sdf, "sdf");
        
        // Color mapping based on selected endpoint
        const activeEndpoint = attrSelect.value;
        const attributions = currentMoleculeData.attributions[activeEndpoint] || [];
        
        // Style and color atoms individually based on GDL Integrated Gradients
        let model = glViewer.getModel(0);
        let atoms = model.selectedAtoms({});
        
        atoms.forEach((atom, index) => {
            const attrScore = attributions[index] !== undefined ? attributions[index] : 0.0;
            const hexColor = getAttributionColor(attrScore);
            
            // Set individual atom styles (spheres for nodes, sticks for bonds)
            glViewer.setStyle(
                { serial: atom.serial },
                {
                    sphere: { color: hexColor, scale: 0.32 },
                    stick: { color: hexColor, radius: 0.16 }
                }
            );
        });
        
        // Zoom and spin conformer automatically
        glViewer.zoomTo();
        glViewer.render();
        glViewer.animate({ loop: "backward", step: 0.3 });
    }

    // Process Molecular Analysis via Flask Backend
    async function runAnalysis(smiles) {
        if (!smiles) return;
        
        // Show loading state
        loadingOverlay.classList.add("active");
        
        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ smiles })
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                alert(data.error || "Analysis failed.");
                return;
            }
            
            currentMoleculeData = data;
            
            // 1. Update Physical Descriptors Panel
            valFormula.innerText = data.formula;
            valMW.innerText = data.mw;
            valLogP.innerText = data.logp;
            valTPSA.innerText = data.tpsa;
            valHBD.innerText = data.hbd;
            valHBA.innerText = data.hba;
            propsPanel.style.display = "block";
            
            // 2. Render 3D Conformer Heatmap
            render3DConformer();
            
            // 3. Update Toxicity gauges & meters
            for (const endpoint in scores) {
                const prob = data.predictions[endpoint];
                const scorePercent = (prob * 100).toFixed(1) + "%";
                
                // Update score labels
                scores[endpoint].text.innerText = scorePercent;
                scores[endpoint].text.style.color = prob > 0.6 ? "var(--toxic)" : (prob >= 0.3 ? "var(--warning)" : "var(--safe)");
                
                // Update progress fills
                const fillElement = scores[endpoint].fill;
                fillElement.className = "progress-fill " + getThemeClass(prob);
                fillElement.style.width = scorePercent;
            }
            
            // 4. Update Heuristic Advice (Left panel)
            copilotBody.innerHTML = "";
            data.advice.forEach(tip => {
                const bullet = document.createElement("div");
                bullet.className = "copilot-bullet";
                
                // Simple Markdown bold converter for advice texts
                let htmlTip = tip.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
                bullet.innerHTML = htmlTip;
                copilotBody.appendChild(bullet);
            });
            
            // 5. Update Generative MPO Analogs Cards (Right panel)
            const analogsContainer = document.getElementById("analogsContainer");
            analogsContainer.innerHTML = "";
            
            if (data.analogs && data.analogs.length > 0) {
                data.analogs.forEach(analog => {
                    // Determine SA badge theme
                    let saClass = "easy";
                    let saLabel = "Easy";
                    if (analog.sa_score > 6.0) {
                        saClass = "difficult";
                        saLabel = "Hard";
                    } else if (analog.sa_score > 3.5) {
                        saClass = "moderate";
                        saLabel = "Moderate";
                    }
                    
                    const card = document.createElement("div");
                    card.className = "analog-card";
                    
                    // Format advice text (handling Markdown bold)
                    let htmlChange = analog.change.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
                    
                    // Create comparison table rows for physical properties and toxicities
                    const properties = [
                        { name: "hERG Cardiotox", parent: data.predictions.herg, analog: analog.predictions.herg, format: "prob" },
                        { name: "Ames Genotox", parent: data.predictions.ames, analog: analog.predictions.ames, format: "prob" },
                        { name: "DILI Hepatotox", parent: data.predictions.dili, analog: analog.predictions.dili, format: "prob" },
                        { name: "Acute Tox (LD50)", parent: data.predictions.ld50, analog: analog.predictions.ld50, format: "prob" },
                        { name: "LogP Lipophilicity", parent: data.logp, analog: analog.logp, format: "num" },
                        { name: "Mol Weight (g/mol)", parent: data.mw, analog: analog.mw, format: "num" }
                    ];
                    
                    let tableRows = "";
                    properties.forEach(prop => {
                        let parentVal = prop.parent;
                        let analogVal = prop.analog;
                        
                        let parentStr = prop.format === "prob" ? (parentVal * 100).toFixed(1) + "%" : parentVal.toString();
                        let analogStr = prop.format === "prob" ? (analogVal * 100).toFixed(1) + "%" : analogVal.toString();
                        
                        // Calculate change delta and class (green = improvement, red = deterioration, grey = neutral)
                        let deltaClass = "neutral";
                        let deltaIcon = "→";
                        
                        if (prop.format === "prob") {
                            // Toxicity: lower is better
                            if (analogVal < parentVal - 0.05) {
                                deltaClass = "positive"; // Green (improved safety)
                                deltaIcon = "↓ Safe";
                            } else if (analogVal > parentVal + 0.05) {
                                deltaClass = "negative"; // Red (added toxicity risk)
                                deltaIcon = "↑ Risk";
                            }
                        } else if (prop.name.startsWith("LogP")) {
                            // Lower LogP is generally safer
                            if (analogVal < parentVal) {
                                deltaClass = "positive";
                                deltaIcon = "↓";
                            } else if (analogVal > parentVal) {
                                deltaClass = "negative";
                                deltaIcon = "↑";
                            }
                        } else if (prop.name.startsWith("Mol Weight")) {
                            // Lower Molecular Weight is generally cleaner for synth/oral absorption
                            if (analogVal < parentVal) {
                                deltaClass = "positive";
                                deltaIcon = "↓";
                            } else if (analogVal > parentVal) {
                                deltaClass = "negative";
                                deltaIcon = "↑";
                            }
                        }
                        
                        tableRows += `
                            <tr>
                                <td style="font-weight: 600; color: var(--text-primary);">${prop.name}</td>
                                <td>${parentStr}</td>
                                <td>${analogStr}</td>
                                <td><span class="mpo-diff ${deltaClass}">${deltaIcon}</span></td>
                            </tr>
                        `;
                    });
                    
                    card.innerHTML = `
                        <div class="analog-card-header">
                            <div class="analog-change-title">${htmlChange}</div>
                            <span class="sa-badge ${saClass}">SA: ${analog.sa_score.toFixed(1)} (${saLabel})</span>
                        </div>
                        
                        <table class="mpo-table">
                            <thead>
                                <tr>
                                    <th>Parameter</th>
                                    <th>Parent</th>
                                    <th>Analog</th>
                                    <th>Shift</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${tableRows}
                            </tbody>
                        </table>
                        
                        <div class="analog-actions">
                            <button class="smiles-btn" data-smiles="${analog.smiles}">
                                📋 Copy SMILES
                            </button>
                            <button class="load-btn" data-smiles="${analog.smiles}">
                                ⚡ Load Analog
                            </button>
                        </div>
                    `;
                    
                    // Bind events inside the card
                    const copyBtn = card.querySelector(".smiles-btn");
                    copyBtn.addEventListener("click", (e) => {
                        e.stopPropagation();
                        navigator.clipboard.writeText(analog.smiles);
                        copyBtn.innerText = "✓ Copied!";
                        setTimeout(() => {
                            copyBtn.innerText = "📋 Copy SMILES";
                        }, 1500);
                    });
                    
                    const loadBtn = card.querySelector(".load-btn");
                    loadBtn.addEventListener("click", (e) => {
                        e.stopPropagation();
                        smilesInput.value = analog.smiles;
                        runAnalysis(analog.smiles);
                    });
                    
                    analogsContainer.appendChild(card);
                });
            } else {
                analogsContainer.innerHTML = `
                    <div class="empty-state-text" style="position: static; transform: none; width: 100%; color: var(--text-secondary); padding: 25px; font-size: 0.85rem; line-height: 1.5; border: 1px dashed var(--border-glass); border-radius: 12px; text-align: center;">
                        This molecule has a clean safety profile! No mutagenic/cardiotoxic liabilities were flagged by the GNN model, so no bioisosteric modifications are necessary.
                    </div>
                `;
            }
            
            copilotPanel.style.display = "block";
            
        } catch (err) {
            console.error(err);
            alert("App Error: " + err.message + "\n\nIf it is a network connection error, please verify the Flask backend is active at http://127.0.0.1:5000");
        } finally {
            loadingOverlay.classList.remove("active");
        }
    }

    // Trigger analysis
    analyzeBtn.addEventListener("click", () => {
        const smiles = smilesInput.value.trim();
        if (smiles) {
            runAnalysis(smiles);
        } else {
            alert("Please enter a valid SMILES string.");
        }
    });

    // Preset Badges Click Handler
    presetBadges.forEach(badge => {
        badge.addEventListener("click", () => {
            const smiles = badge.getAttribute("data-smiles");
            smilesInput.value = smiles;
            runAnalysis(smiles);
        });
    });

    // Dynamic Color Map Selection Change
    attrSelect.addEventListener("change", () => {
        if (currentMoleculeData) {
            render3DConformer();
        }
    });
});
