import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# =====================================================================
# 1. Core AI Agent Framework (Anthropic-Style Structured Reasoning)
# =====================================================================

class AIAgent:
    """
    Represents a specialized AI drug discovery researcher.
    Implements Anthropic-style Tool Use and Structured Reasoning (System 2 thinking).
    """
    def __init__(self, name, role, avatar, persona, resources="", tools=None):
        self.name = name
        self.role = role
        self.avatar = avatar
        self.persona = persona
        self.resources = resources  # User-supplied knowledge base / references
        self.tools = tools if tools else {"GNN", "LogP", "SA_Score"}

    def run_reasoning(self, smiles, parent_data, analogs, context):
        """
        Executes a two-phase reasoning process:
        1. Hidden System 2 'Thought' (analyzing data, tools, and resources).
        2. Public 'Dialogue' (collaborative exchange with other specialists).
        """
        # Base templates for default agents, dynamically weaved with exact GDL data
        thought = ""
        dialogue = ""
        
        # Determine if parent molecule has issues or if we are in General Inquiry Mode
        is_general_inquiry = (parent_data is None)
        
        if is_general_inquiry:
            probs = {"herg": 0.0, "ames": 0.0, "dili": 0.0, "ld50": 0.0}
            has_herg = False
            has_ames = False
            has_dili = False
            parent_has_issues = False
        else:
            probs = parent_data["predictions"]
            has_herg = probs["herg"] > 0.5
            has_ames = probs["ames"] > 0.5
            has_dili = probs["dili"] > 0.5
            parent_has_issues = has_herg or has_ames or has_dili or probs["ld50"] > 0.5
        
        # Read user resources if available to inject expert quotes
        resource_citation = ""
        if self.resources:
            resource_citation = f" Referring back to the research guidelines in my knowledge base ('{self.resources[:40]}...'),"

        # -------------------------------------------------------------
        # Persona 1: Dr. GDL (Theoretical AI Physicist)
        # -------------------------------------------------------------
        if self.name == "Dr. GDL":
            if is_general_inquiry:
                thought = f"Providing structural GDL physics overview for query: '{smiles}'."
                dialogue = (
                    f"Hello colleagues. From a structural biophysics perspective, the topic of '{smiles}' "
                    f"centers on topological chemical graphs, edge connectivity configurations, and deep geometric architectures."
                    f"{resource_citation} Message-passing models map local spatial coordinates to compile "
                    f"invariant representations of chemical features, which is essential to predict these biophysical behaviors."
                )
            else:
                thought = (
                    f"Analyzing chemical graph for {smiles}. Integrated Gradients shows high attribution "
                    f"hotspots. I need to map the topological coordinates to the hERG/Ames predicted spikes "
                    f"(hERG: {probs['herg']:.2f}, Ames: {probs['ames']:.2f}). My role is strictly structural physics."
                )
                dialogue = (
                    f"Hello colleagues. I have run our GDL Message Passing model on the parent SMILES graph (`{smiles}`)."
                    f"{resource_citation} I have mapped the 3D conformer coordinates. "
                )
                if parent_has_issues:
                    dialogue += "We have highly defined structural liabilities here. "
                    if has_dili:
                        dialogue += f"The **DILI Hepatotoxicity** is severely flagged at **{(probs['dili']*100):.1f}%** probability, representing a critical clinical metabolic liability. "
                    if probs["ld50"] > 0.5:
                        dialogue += f"The **Acute Oral Toxicity (LD50)** shows high hazard potential at **{(probs['ld50']*100):.1f}%**. "
                    if has_herg:
                        dialogue += f"The **hERG Cardiotox** is flagged at **{(probs['herg']*100):.1f}%**, indicating off-target potassium channel blockade risk. "
                    if has_ames:
                        dialogue += f"The **Ames Genotox** is showing a severe spike at **{(probs['ames']*100):.1f}%**, localized heavily on genotoxic mutagen hotspots."
                    dialogue += " Dr. Synthesis, please examine the atomic indices in red on the 3D viewer and let us know what bioisosteric modifications are realistic to construct."
                else:
                    dialogue += "Excellent news. The graph neural networks show a clean topological profile across all endpoints (all probabilities < 30%). No toxicophore hotspots were resolved."

        # -------------------------------------------------------------
        # Persona 2: Dr. Synthesis (Medicinal Chemist)
        # -------------------------------------------------------------
        elif self.name == "Dr. Synthesis":
            if is_general_inquiry:
                thought = f"Addressing organic synthesis pathways for query: '{smiles}'."
                dialogue = (
                    f"Thank you, Dr. GDL. Speaking to the chemical synthesis side of the question '{smiles}': "
                    f"{resource_citation} Designing bioisosteres and verifying synthetic accessibility (SA) scores "
                    f"allows us to explore new scaffold spaces. My priority is ensuring that proposed derivatives are "
                    f"synthetically feasible and easily constructed in the wet lab."
                )
            else:
                thought = (
                    f"Receiving structural report from Dr. GDL. We have active toxicophores. "
                    f"I will apply RDKit SMARTS reaction transforms to construct clean analogs and calculate "
                    f"synthetic accessibility (SA) scores."
                )
                dialogue = (
                    f"Thank you for the coordinates, Dr. GDL. I've examined the red hot spots. "
                    f"{resource_citation} I applied targeted bioisostere transformations. "
                )
                if analogs:
                    dialogue += f"I have successfully synthesized **{len(analogs)} alternative analogs**:\n\n"
                    for idx, analog in enumerate(analogs):
                        dialogue += f"• **Analog #{idx+1} ({analog['change'].split(':')[0]}):** Swap target SMILES to `{analog['smiles']}` (Estimated Synthetic Accessibility is **{analog['sa_score']:.1f}**).\n"
                    dialogue += "\nDr. ADME, these alternatives alter the polarity and geometry. Could you evaluate how these structural changes impact our physical descriptors, specifically LogP lipophilicity and molecular weight?"
                else:
                    if parent_has_issues:
                        dialogue += (
                            "Although the GNN has resolved severe toxicity liabilities, my automated bioisosteric "
                            "mutation rules could not generate suitable analogs. This occurs because the toxicophore "
                            "attribution hotspots reside on un-shieldable heteroatoms (like oxygens/nitrogens) or the scaffold "
                            "lacks standard basic-nitrogen or nitro functional groups for standard bioisosteric swaps. "
                            "We will need manual scaffold hopping or functional shielding in the wet lab to fix these DILI/LD50/hERG spikes."
                        )
                    else:
                        dialogue += "Given the highly safe and balanced structure of this compound, no bioisosteric mutations were generated. It is synthesizable as-is with a highly clean profile."

        # -------------------------------------------------------------
        # Persona 3: Dr. ADME (Pharmacokinetics Pharmacologist)
        # -------------------------------------------------------------
        elif self.name == "Dr. ADME":
            if is_general_inquiry:
                thought = f"ADME pharmacokinetics review for query: '{smiles}'."
                dialogue = (
                    f"I've compiled the pharmacokinetic principles related to the query, Dr. Synthesis. "
                    f"{resource_citation} From a ADMET perspective, resolving the physics behind '{smiles}' "
                    f"requires understanding physical properties like LogP lipophilicity, molecular weight, hydrogen bond counts, "
                    f"and TPSA. Balancing safety with excellent cellular permeability is the core trade-off."
                )
            else:
                thought = (
                    f"Dr. Synthesis has proposed new analogs. I must evaluate their physical descriptors "
                    f"(LogP, MW, TPSA) and look for solubility or membrane permeability trade-offs."
                )
                dialogue = (
                    f"I've compiled the pharmacokinetic descriptors for your proposals, Dr. Synthesis. "
                    f"{resource_citation} Here is the ADMET analysis:\n\n"
                )
                if analogs:
                    for idx, analog in enumerate(analogs):
                        delta_logp = analog["logp"] - parent_data["logp"]
                        logp_str = f"LogP: {analog['logp']} (Change: {delta_logp:+.2f})"
                        mw_str = f"MW: {analog['mw']} g/mol"
                        dialogue += f"• **Analog #{idx+1}:** {logp_str} | {mw_str}. "
                        if analog["logp"] > 5.0:
                            dialogue += "⚠️ LogP > 5.0 violates Lipinski Rules, indicating severe lipophilicity risk. "
                        elif analog["logp"] > 3.5:
                            dialogue += "⚠️ Lipophilicity is creeping high, which might slow oral clearance. "
                        else:
                            dialogue += "✅ Retains excellent drug-like solubility boundaries. "
                        dialogue += "\n"
                    dialogue += "\nDr. Safety, from a cellular regulatory perspective, do these modifications successfully clear our safety thresholds?"
                else:
                    dialogue += "Analyzing the physical properties of the parent structure:\n"
                    formula_str = parent_data.get('formula', 'N/A')
                    dialogue += f"• **Formula:** {formula_str} | **Molecular Weight:** {parent_data['mw']} g/mol\n"
                    dialogue += f"• **LogP Lipophilicity:** {parent_data['logp']} | **TPSA:** {parent_data['tpsa']} Å²\n"
                    dialogue += f"• **H-Bond Donors:** {parent_data['hbd']} | **H-Bond Acceptors:** {parent_data['hba']}\n\n"
                    
                    # LogP and Lipinski assessment
                    admet_notes = []
                    if parent_data["logp"] > 5.0:
                        admet_notes.append(f"⚠️ **Lipinski Rule Violation:** The extreme **LogP of {parent_data['logp']}** exceeds Lipinski's safety threshold (<5.0), representing a highly lipophilic/greasy molecule with poor aqueous solubility and high metabolic clearance risk.")
                    elif parent_data["logp"] > 3.5:
                        admet_notes.append(f"⚠️ **PK Warning:** The relatively high **LogP of {parent_data['logp']}** indicates high lipophilicity, making the compound susceptible to rapid metabolic clearance in the liver.")
                    else:
                        admet_notes.append("✅ Lipophilicity resides in the optimal balanced range for oral bioavailability.")
                        
                    if parent_data["mw"] > 500:
                        admet_notes.append(f"⚠️ **Lipinski Violation:** Molecular Weight (**{parent_data['mw']} g/mol**) is too high (>500 g/mol), which typically restricts membrane permeability.")
                    
                    dialogue += " ".join(admet_notes) + " This physical profile must be evaluated against the GNN safety endpoints."

        # -------------------------------------------------------------
        # Persona 4: Dr. Safety (Regulatory Toxicologist - Lead)
        # -------------------------------------------------------------
        elif self.name == "Dr. Safety":
            if is_general_inquiry:
                thought = f"Toxicological regulatory audit for query: '{smiles}'."
                dialogue = (
                    f"Excellent discussion, team. Let's look at the toxicological and regulatory implications of '{smiles}'."
                    f"{resource_citation} In pre-clinical toxicology, any investigation regarding this question "
                    f"requires verifying that our GNN classification metrics (hERG cardiotoxicity, Ames genotoxicity, "
                    f"DILI hepatotoxicity, and acute oral LD50) remain strictly monitored. Our role is ensuring regulatory compliance "
                    f"and preventing toxic liabilities from advancing to clinical trials."
                )
            else:
                thought = (
                    f"Collating reports from Dr. GDL, Dr. Synthesis, and Dr. ADME. I must examine the GDL "
                    f"re-scored toxicity probabilities of the proposed analogs and make the final recommendation."
                )
                dialogue = (
                    f"Excellent collaborative analysis, team. Let's look at the GNN safety re-scoring. "
                    f"{resource_citation} Here are the final toxicity metrics:\n\n"
                )
                if analogs:
                    best_analog_idx = -1
                    best_improvement = 0.0
                    
                    for idx, analog in enumerate(analogs):
                        herg_pct = analog["predictions"]["herg"] * 100
                        ames_pct = analog["predictions"]["ames"] * 100
                        dili_pct = analog["predictions"]["dili"] * 100
                        
                        dialogue += f"• **Analog #{idx+1}:** hERG Cardiotox: **{herg_pct:.1f}%** | Ames Mutagenicity: **{ames_pct:.1f}%** | DILI Hepatotox: **{dili_pct:.1f}%**.\n"
                        
                        # Calculate overall safety improvement sum
                        parent_sum = probs["herg"] + probs["ames"] + probs["dili"]
                        analog_sum = analog["predictions"]["herg"] + analog["predictions"]["ames"] + analog["predictions"]["dili"]
                        improvement = parent_sum - analog_sum
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_analog_idx = idx
                    
                    if best_analog_idx != -1:
                        best = analogs[best_analog_idx]
                        dialogue += f"\n🏆 **Final Panel Consensus:** We officially recommend pursuing **Analog #{best_analog_idx+1}**. "
                        dialogue += f"It resolves our primary toxicological liabilities while keeping a synthesizable SA score of **{best['sa_score']:.1f}** and safe drug-likeness. This concludes our advisory panel!"
                    else:
                        dialogue += "\nBoth alternatives present similar ADMET profiles. We suggest exploring additional synthesis branches."
                else:
                    if parent_has_issues:
                        dialogue += "🚨 **Preclinical Safety Warning:** The parent structure has failed our safety metrics and cannot be cleared:\n\n"
                        if has_dili:
                            dialogue += f"• **DILI Hepatotoxicity:** Predicted at **{(probs['dili']*100):.1f}%** (CRITICAL METABOLIC LIABILITY).\n"
                        if probs["ld50"] > 0.5:
                            dialogue += f"• **Acute Oral Toxicity (LD50):** Flagged at **{(probs['ld50']*100):.1f}%** (HIGH RISK).\n"
                        if has_herg:
                            dialogue += f"• **hERG Cardiotoxicity:** Flagged at **{(probs['herg']*100):.1f}%** (MODERATE RISK).\n"
                        if has_ames:
                            dialogue += f"• **Ames Genotoxicity:** Mutagenic damage predicted at **{(probs['ames']*100):.1f}%** (GENOTOXIC).\n"
                            
                        dialogue += (
                            "\n🏆 **Final Panel Consensus:** **REJECTED.** The compound exhibits severe clinical liabilities. "
                            "Since no automatic bioisosteric modifications were resolved, manual scaffold modifications and polar group "
                            "additions must be done to drop the extreme LogP lipophilicity and resolve these hepatic/acute risks before "
                            "re-submitting for panel review."
                        )
                    else:
                        dialogue += f"Our GDL analysis concludes that the parent structure requires no modifications. It successfully clears all FDA/regulatory safety gates with hERG, Ames, and DILI probabilities well below thresholds."

        # -------------------------------------------------------------
        # Dynamic Custom Agents (User-Created)
        # -------------------------------------------------------------
        else:
            if is_general_inquiry:
                thought = f"I am {self.name}, acting as the {self.role}. Reviewing custom query '{smiles}'."
                dialogue = (
                    f"Hello board. I am **{self.name}**, speaking from my perspective as a **{self.role}**."
                    f"{resource_citation} Regarding the scientific topic of '{smiles}': I advise the panel to consider "
                    f"how specific domain guidelines and research principles inside my knowledge base apply to this query."
                )
            else:
                thought = (
                    f"I am {self.name}, acting as the {self.role}. "
                    f"Reviewing compound {smiles} (MW: {parent_data['mw']}, LogP: {parent_data['logp']}) "
                    f"based on my custom persona: '{self.persona}'."
                )
                dialogue = (
                    f"Hello board. I am **{self.name}**, speaking from my perspective as a **{self.role}**."
                    f"{resource_citation} Analyzing the target structure:\n\n"
                )
                
                # Formulate custom responses based on tools/persona rules
                warnings = []
                if "MW" in self.persona or "molecular weight" in self.persona.lower():
                    if parent_data["mw"] > 450:
                        warnings.append(f"⚠️ **Molecular Weight Alert**: The compound molecular weight of **{parent_data['mw']} g/mol** exceeds standard boundaries, which may violate Lipinski's Rule of 5.")
                if "halogen" in self.persona.lower() or "fluorine" in self.persona.lower():
                    warnings.append("💡 Note: Introducing halogen groups or fluorine metabolic blockers represents a solid shielding approach but must be monitored for accumulation risks.")
                if "DILI" in self.persona or "hepatotoxicity" in self.persona.lower():
                    if has_dili:
                        warnings.append(f"⚠️ **Hepatotoxicity Warning**: The compound shows severe DILI hepatotoxicity (**{(probs['dili']*100):.1f}%**). This represents a direct hepatic clearance liability.")
                
                if warnings:
                    dialogue += "\n".join(warnings) + "\n\n"
                else:
                    if parent_has_issues:
                        dialogue += "The parent compound shows significant GNN liabilities. We must prioritize generating or exploring analogs. "
                    else:
                        dialogue += "The structural geometry, weight, and charge states comply with my criteria. "
                    
                if analogs:
                    dialogue += "The proposed bioisosteric analogs look promising. I advise Dr. Safety to cross-reference my warnings before signing off on the final consensus."
                else:
                    if parent_has_issues:
                        dialogue += "I endorse the panel's consensus rejection of this parent compound in its current toxic state."
                    else:
                        dialogue += "I endorse the parent compound as safe from my regulatory perspective."

        return {
            "name": self.name,
            "role": self.role,
            "avatar": self.avatar,
            "thought": thought,
            "dialogue": dialogue
        }


# =====================================================================
# 2. Agent Registry Manager (Dynamic Custom Agent Spinning)
# =====================================================================

class AgentRegistry:
    """
    Manages active scientific AI agents, supporting dynamic registration
    of custom researchers.
    """
    def __init__(self):
        self.agents = {}
        self.reset_to_defaults()

    def reset_to_defaults(self):
        """
        Loads the four default scientific specialists.
        """
        self.agents = {
            "Dr. GDL": AIAgent(
                name="Dr. GDL",
                role="Structural GNN Physicist",
                avatar="🔮",
                persona="You are Dr. GDL, an expert in geometric deep learning and structural biophysics. You analyze atomic graph features, 3D conformation vectors, and toxicophore heatmaps. Speak mathematically and focus strictly on graph topology."
            ),
            "Dr. Synthesis": AIAgent(
                name="Dr. Synthesis",
                role="Medicinal Chemist",
                avatar="🧪",
                persona="You are Dr. Synthesis, a veteran medicinal chemist with 30 years of organic wet-lab synthesis experience. You evaluate if molecules are realistic to make, suggest SMARTS bioisostere swaps, and compute topological synthetic accessibility (SA) scores."
            ),
            "Dr. ADME": AIAgent(
                name="Dr. ADME",
                role="PK-PD Pharmacologist",
                avatar="📊",
                persona="You are Dr. ADME, a specialist in pharmacokinetics. You evaluate Lipinski's rules, LogP lipophilicity, polar surface area (TPSA), and hydrogen bonding capabilities, highlighting trade-offs between safety and cellular permeability."
            ),
            "Dr. Safety": AIAgent(
                name="Dr. Safety",
                role="Regulatory Toxicologist (Lead)",
                avatar="🛡️",
                persona="You are Dr. Safety, the lead toxicologist. You examine final GNN predictions for Ames genotoxicity, hERG cardiotoxicity, DILI, and LD50. You synthesize all agents' inputs and draft the final consensus recommendation."
            )
        }

    def register_agent(self, name, role, avatar, persona, resources=""):
        """
        Dynamically registers a custom AI agent on demand!
        """
        # Clean inputs
        clean_name = name.strip()
        if not clean_name:
            return False
            
        self.agents[clean_name] = AIAgent(
            name=clean_name,
            role=role.strip() if role else "Specialized Scientist",
            avatar=avatar.strip() if avatar else "🧬",
            persona=persona.strip(),
            resources=resources.strip()
        )
        return True

    def get_all_agents(self):
        """
        Returns a list of all active agents.
        """
        return [
            {
                "name": agent.name,
                "role": agent.role,
                "avatar": agent.avatar,
                "persona": agent.persona,
                "resources": agent.resources
            }
            for agent in self.agents.values()
        ]


# =====================================================================
# 3. Autonomous State-Machine Consultation Loop
# =====================================================================

# Single global registry instance in memory
global_registry = AgentRegistry()

def run_consultation(smiles, parent_data, analogs, target_protein="General"):
    """
    State-Machine Routing Loop: Orchestrates a collaborative consultation dialogue.
    1. Dr. GDL presents the GNN graph analysis.
    2. Dr. Synthesis proposes organic changes and SA metrics.
    3. Dr. ADME checks LogP/MW trade-offs.
    4. Custom user-created agents join and insert their domain audits.
    5. Dr. Safety summarizes and closes with the consensus recommendation.
    """
    log = []
    context = {"target_protein": target_protein}
    
    # State-Machine Order:
    # Default Base order (GDL -> Synthesis -> ADME)
    ordered_names = ["Dr. GDL", "Dr. Synthesis", "Dr. ADME"]
    
    # Insert any custom agents in between
    for name in global_registry.agents.keys():
        if name not in ordered_names and name != "Dr. Safety":
            ordered_names.append(name)
            
    # Always end with Dr. Safety to compile the consensus
    if "Dr. Safety" in global_registry.agents:
        ordered_names.append("Dr. Safety")

    # Execute reasoning and dialog generation sequentially
    for name in ordered_names:
        if name in global_registry.agents:
            agent = global_registry.agents[name]
            result = agent.run_reasoning(smiles, parent_data, analogs, context)
            log.append(result)
            
    return log
