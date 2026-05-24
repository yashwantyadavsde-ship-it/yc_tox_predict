// ToxGDL Client Controller - 3D Visualizer, Multi-Agent Boardroom and API Interface
document.addEventListener("DOMContentLoaded", () => {
    // UI Elements
    const smilesInput = document.getElementById("smilesInput");
    const targetSelect = document.getElementById("targetSelect");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const presetBadges = document.querySelectorAll(".preset-badge");
    const loadingOverlay = document.getElementById("loadingOverlay");
    
    // Properties Panel
    const propsPanel = document.getElementById("propsPanel");
    const valFormula = document.getElementById("valFormula");
    const valMW = document.getElementById("valMW");
    const valLogP = document.getElementById("valLogP");
    const valTPSA = document.getElementById("valTPSA");
    const valHBD = document.getElementById("valHBD");
    const valHBA = document.getElementById("valHBA");
    
    // Isomorphic Docking Affinity Display Panel
    const dockingAffinityCard = document.getElementById("dockingAffinityCard");
    const dockingEnergy = document.getElementById("dockingEnergy");
    const dockingKd = document.getElementById("dockingKd");
    const dockingTargetName = document.getElementById("dockingTargetName");
    
    // Tab Elements
    const tabBoardroomBtn = document.getElementById("tabBoardroomBtn");
    const tabCustomAgentBtn = document.getElementById("tabCustomAgentBtn");
    const tabBoardroom = document.getElementById("tabBoardroom");
    const tabCustomAgent = document.getElementById("tabCustomAgent");
    
    // Custom Agent Builder Inputs
    const agentNameInput = document.getElementById("agentName");
    const agentAvatarInput = document.getElementById("agentAvatar");
    const agentRoleInput = document.getElementById("agentRole");
    const agentPersonaInput = document.getElementById("agentPersona");
    const agentResourcesInput = document.getElementById("agentResources");
    const deployAgentBtn = document.getElementById("deployAgentBtn");
    
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
    const boardroomThread = document.getElementById("boardroomThread");
    const analogsContainer = document.getElementById("analogsContainer");
    
    // Active prediction data store
    let currentMoleculeData = null;
    let glViewer = null;

    // Helper: Map Integrated Gradients score [-1.0, 1.0] to a stunning RGB color
    function getAttributionColor(score) {
        if (score > 0) {
            // Map positive score to deep red
            const r = 255;
            const g = Math.round(255 - score * 187);
            const b = Math.round(255 - score * 187);
            return `rgb(${r}, ${g}, ${b})`;
        } else if (score < 0) {
            // Map negative score to green
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
        viewerElement.innerHTML = "";
        
        if (typeof $3Dmol === "undefined") {
            viewerElement.innerHTML = `<div class="empty-state-text" style="color: var(--toxic); font-weight: 600; padding: 20px; line-height: 1.6;">
                ⚠️ 3Dmol.js Library Error:<br>
                Unable to render 3D stereochemical conformer because the 3Dmol library failed to load in your browser.<br><br>
                Please verify your internet connection or check your browser console for script blocker extensions.
            </div>`;
            return;
        }
        
        glViewer = $3Dmol.createViewer(viewerElement, { backgroundColor: "#060813" });
        glViewer.addModel(currentMoleculeData.sdf, "sdf");
        
        // Color mapping based on selected endpoint
        const activeEndpoint = attrSelect.value;
        const attributions = currentMoleculeData.attributions[activeEndpoint] || [];
        
        let model = glViewer.getModel(0);
        let atoms = model.selectedAtoms({});
        
        atoms.forEach((atom, index) => {
            const attrScore = attributions[index] !== undefined ? attributions[index] : 0.0;
            const hexColor = getAttributionColor(attrScore);
            
            glViewer.setStyle(
                { serial: atom.serial },
                {
                    sphere: { color: hexColor, scale: 0.32 },
                    stick: { color: hexColor, radius: 0.16 }
                }
            );
        });
        
        glViewer.zoomTo();
        glViewer.render();
        glViewer.animate({ loop: "backward", step: 0.3 });
    }

    // Process Molecular Analysis via Flask Backend
    async function runAnalysis(smiles) {
        if (!smiles) return;
        
        loadingOverlay.classList.add("active");
        const target = targetSelect.value;
        
        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ smiles, target })
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
            
            // 2. Render Isomorphic 3D Pocket Affinity Card
            if (data.affinity) {
                dockingEnergy.innerText = data.affinity.delta_g.toFixed(2) + " kcal/mol";
                let kdVal = data.affinity.kd_nm;
                let kdStr = kdVal >= 1000 ? (kdVal / 1000).toFixed(1) + " μM" : kdVal.toFixed(1) + " nM";
                dockingKd.innerText = "Kd: " + kdStr;
                dockingTargetName.innerText = "Target Pocket: " + data.affinity.target;
                dockingAffinityCard.style.display = "block";
            } else {
                dockingAffinityCard.style.display = "none";
            }
            
            // 3. Render 3D Conformer Heatmap
            render3DConformer();
            
            // 4. Update Toxicity gauges & meters
            for (const endpoint in scores) {
                const prob = data.predictions[endpoint];
                const scorePercent = (prob * 100).toFixed(1) + "%";
                
                scores[endpoint].text.innerText = scorePercent;
                scores[endpoint].text.style.color = prob > 0.6 ? "var(--toxic)" : (prob >= 0.3 ? "var(--warning)" : "var(--safe)");
                
                const fillElement = scores[endpoint].fill;
                fillElement.className = "progress-fill " + getThemeClass(prob);
                fillElement.style.width = scorePercent;
            }
            
            // 5. Update AI Research Boardroom (Left Panel Tab 1)
            boardroomThread.innerHTML = "";
            if (data.consultation_log && data.consultation_log.length > 0) {
                data.consultation_log.forEach((logItem, index) => {
                    let avatarClass = "custom";
                    if (logItem.name === "Dr. GDL") avatarClass = "gdl";
                    else if (logItem.name === "Dr. Synthesis") avatarClass = "synthesis";
                    else if (logItem.name === "Dr. ADME") avatarClass = "adme";
                    else if (logItem.name === "Dr. Safety") avatarClass = "safety";
                    
                    const bubble = document.createElement("div");
                    bubble.className = "chat-bubble";
                    
                    // Simple Markdown bold and inline code formatters
                    let htmlDialogue = logItem.dialogue
                        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
                        .replace(/`([^`]+)`/g, "<code style='background: rgba(255,255,255,0.08); padding: 2px 4px; border-radius: 4px;'>$1</code>")
                        .replace(/\n/g, "<br>");
                    
                    bubble.innerHTML = `
                        <div class="chat-avatar ${avatarClass}">${logItem.avatar}</div>
                        <div class="chat-content-wrapper">
                            <div class="chat-meta">
                                <span class="chat-name">${logItem.name}</span>
                                <span class="chat-role">${logItem.role}</span>
                            </div>
                            <div class="chat-body-bubble">
                                ${htmlDialogue}
                            </div>
                            <button class="thought-toggle" id="toggleThought-${index}">
                                🧠 View System 2 Thought
                            </button>
                            <div class="thought-block" id="thoughtBlock-${index}">
                                ${logItem.thought}
                            </div>
                        </div>
                    `;
                    
                    const toggleBtn = bubble.querySelector(`#toggleThought-${index}`);
                    const thoughtBlock = bubble.querySelector(`#thoughtBlock-${index}`);
                    toggleBtn.addEventListener("click", () => {
                        thoughtBlock.classList.toggle("visible");
                        toggleBtn.innerHTML = thoughtBlock.classList.contains("visible") 
                            ? "🧠 Hide System 2 Thought" 
                            : "🧠 View System 2 Thought";
                    });
                    
                    boardroomThread.appendChild(bubble);
                });
                
                // Auto-scroll boardroom chat to top/start to view introduction
                boardroomThread.scrollTop = 0;
            }
            
            // Switch tabs to courtroom chat thread
            tabBoardroomBtn.click();
            
            // 6. Update Generative MPO Analogs Cards (Right Panel)
            analogsContainer.innerHTML = "";
            if (data.analogs && data.analogs.length > 0) {
                data.analogs.forEach(analog => {
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
                    
                    let htmlChange = analog.change.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
                    
                    // Multi-parameter comparison list (incorporating Isomorphic pocket affinity score)
                    const properties = [
                        { name: "hERG Cardiotox", parent: data.predictions.herg, analog: analog.predictions.herg, format: "prob" },
                        { name: "Ames Genotox", parent: data.predictions.ames, analog: analog.predictions.ames, format: "prob" },
                        { name: "DILI Hepatotox", parent: data.predictions.dili, analog: analog.predictions.dili, format: "prob" },
                        { name: "Acute Tox (LD50)", parent: data.predictions.ld50, analog: analog.predictions.ld50, format: "prob" },
                        { name: "Target Affinity (Kd)", parent: data.affinity.kd_nm, analog: analog.affinity.kd_nm, format: "kd" },
                        { name: "LogP Lipophilicity", parent: data.logp, analog: analog.logp, format: "num" },
                        { name: "Mol Weight (g/mol)", parent: data.mw, analog: analog.mw, format: "num" }
                    ];
                    
                    let tableRows = "";
                    properties.forEach(prop => {
                        let parentVal = prop.parent;
                        let analogVal = prop.analog;
                        
                        let parentStr = prop.format === "prob" ? (parentVal * 100).toFixed(1) + "%" : (prop.format === "kd" ? (parentVal >= 1000 ? (parentVal / 1000).toFixed(1) + " μM" : parentVal.toFixed(1) + " nM") : parentVal.toString());
                        let analogStr = prop.format === "prob" ? (analogVal * 100).toFixed(1) + "%" : (prop.format === "kd" ? (analogVal >= 1000 ? (analogVal / 1000).toFixed(1) + " μM" : analogVal.toFixed(1) + " nM") : analogVal.toString());
                        
                        let deltaClass = "neutral";
                        let deltaIcon = "→";
                        
                        if (prop.format === "prob") {
                            if (analogVal < parentVal - 0.05) {
                                deltaClass = "positive";
                                deltaIcon = "↓ Safe";
                            } else if (analogVal > parentVal + 0.05) {
                                deltaClass = "negative";
                                deltaIcon = "↑ Risk";
                            }
                        } else if (prop.format === "kd") {
                            // Binding constant: lower Kd represents tighter/better binding to target
                            if (analogVal < parentVal) {
                                deltaClass = "positive"; // Green
                                deltaIcon = "↓ Tighter";
                            } else if (analogVal > parentVal) {
                                deltaClass = "negative"; // Red
                                deltaIcon = "↑ Weaker";
                            }
                        } else if (prop.name.startsWith("LogP")) {
                            if (analogVal < parentVal) {
                                deltaClass = "positive";
                                deltaIcon = "↓";
                            } else if (analogVal > parentVal) {
                                deltaClass = "negative";
                                deltaIcon = "↑";
                            }
                        } else if (prop.name.startsWith("Mol Weight")) {
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
            alert("App Error: " + err.message);
        } finally {
            loadingOverlay.classList.remove("active");
        }
    }

    // -------------------------------------------------------------
    // Tab Toggling Action Handlers
    // -------------------------------------------------------------
    tabBoardroomBtn.addEventListener("click", () => {
        tabBoardroomBtn.classList.add("active");
        tabCustomAgentBtn.classList.remove("active");
        tabBoardroom.classList.add("active");
        tabCustomAgent.classList.remove("active");
    });
    
    tabCustomAgentBtn.addEventListener("click", () => {
        tabCustomAgentBtn.classList.add("active");
        tabBoardroomBtn.classList.remove("active");
        tabCustomAgent.classList.add("active");
        tabBoardroom.classList.remove("active");
    });

    // -------------------------------------------------------------
    // Custom Agent Creator Deploy Handler
    // -------------------------------------------------------------
    deployAgentBtn.addEventListener("click", async () => {
        const name = agentNameInput.value.trim();
        const avatar = agentAvatarInput.value.trim();
        const role = agentRoleInput.value.trim();
        const persona = agentPersonaInput.value.trim();
        const resources = agentResourcesInput.value.trim();
        
        if (!name || !persona) {
            alert("Please fill in the Name and Persona/Philosophy fields for your AI Researcher.");
            return;
        }
        
        loadingOverlay.classList.add("active");
        
        try {
            const response = await fetch("/create_agent", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name, avatar, role, persona, resources })
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                alert(data.error || "Failed to register custom agent.");
                return;
            }
            
            // Notify success
            alert(`🎉 Success! Agent '${name}' has been successfully deployed. Let's run an analysis to let them debate!`);
            
            // Reset inputs
            agentNameInput.value = "";
            agentAvatarInput.value = "🧬";
            agentRoleInput.value = "";
            agentPersonaInput.value = "";
            agentResourcesInput.value = "";
            
            // Switch tabs back to active Board Room boardroom thread
            tabBoardroomBtn.click();
            
            // Re-run analysis if there's an active SMILES string to dynamically include the new agent
            const smiles = smilesInput.value.trim();
            if (smiles) {
                runAnalysis(smiles);
            }
            
        } catch (err) {
            console.error(err);
            alert("Connection Error: " + err.message);
        } finally {
            loadingOverlay.classList.remove("active");
        }
    });

    // -------------------------------------------------------------
    // Global Event Handlers
    // -------------------------------------------------------------
    analyzeBtn.addEventListener("click", () => {
        const smiles = smilesInput.value.trim();
        if (smiles) {
            runAnalysis(smiles);
        } else {
            alert("Please enter a valid SMILES string.");
        }
    });

    presetBadges.forEach(badge => {
        badge.addEventListener("click", () => {
            const smiles = badge.getAttribute("data-smiles");
            smilesInput.value = smiles;
            runAnalysis(smiles);
        });
    });

    attrSelect.addEventListener("change", () => {
        if (currentMoleculeData) {
            render3DConformer();
        }
    });
});
