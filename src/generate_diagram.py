import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_diagram(output_path='docs/workflow_diagram.png'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create figure with dark theme style
    fig, ax = plt.subplots(figsize=(12, 7.5), facecolor='#0f1115')
    ax.set_facecolor('#0f1115')
    
    # Hide axes
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    # Set up styling colors
    color_db = '#3182ce'      # Blue for databases
    color_process = '#805ad5' # Purple for processes
    color_model = '#319795'   # Teal for model layers
    color_decision = '#d69e2e'# Orange/Yellow for decision
    color_output = '#38a169'  # Green for output
    
    text_color = '#ffffff'
    border_color = '#4a5568'
    
    # Helper to draw a box
    def draw_box(x, y, w, h, title, subtitle, color):
        # Draw shadow
        shadow = patches.FancyBboxPatch(
            (x + 0.8, y - 0.8), w, h,
            boxstyle="round,pad=1.5",
            facecolor='#000000', alpha=0.4, zorder=1
        )
        ax.add_patch(shadow)
        
        # Draw box
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=1.5",
            facecolor=color, edgecolor=border_color, linewidth=1.5, zorder=2
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(
            x + w/2, y + h/2 + 1, title,
            color=text_color, fontsize=11, fontweight='bold',
            ha='center', va='center', zorder=3
        )
        ax.text(
            x + w/2, y + h/2 - 2, subtitle,
            color='#cbd5e0', fontsize=8.5,
            ha='center', va='center', zorder=3
        )
        
    # Helper to draw a cylinder (database)
    def draw_cylinder(x, y, w, h, title, subtitle, color):
        draw_box(x, y, w, h, title, subtitle, color)
        
    # Helper to draw arrow
    def draw_arrow(x1, y1, x2, y2, text=""):
        ax.annotate(
            text,
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(facecolor='#a0aec0', edgecolor='#718096', arrowstyle="fancy,head_length=0.4,head_width=0.4,tail_width=0.15", shrinkA=5, shrinkB=5),
            color='#cbd5e0', fontsize=8, ha='center', va='bottom', zorder=4
        )

    # Title of diagram
    ax.text(
        50, 95, "PromoLift - Data & AI Solution Pipeline Architecture",
        color='#ffffff', fontsize=18, fontweight='bold', ha='center', va='center'
    )
    ax.text(
        50, 90, "End-to-End Uplift Modeling & Value-Based Scoring Workflow",
        color='#718096', fontsize=11, ha='center', va='center'
    )

    # 1. Databases (Left)
    draw_cylinder(5, 58, 18, 12, "SME Data Warehouses", "Transactions, Master tables", color_db)
    
    # 2. Feature Engineering
    draw_box(30, 58, 18, 12, "Feature Engineering", "RFM & Promo History", color_process)
    
    # 3. Randomized Holdout (Split)
    draw_box(55, 58, 18, 12, "A/B Pilot Dataset", "80% Treatment / 20% Control", color_model)
    
    # 4. T-Learner (Model T & C)
    draw_box(80, 58, 15, 12, "T-Learner (LightGBM)", "Model_T & Model_C", color_model)
    
    # 5. Customer Scoring (Bottom-Right)
    draw_box(80, 20, 15, 12, "Customer Scoring", "Predict P(T) & P(C)", color_model)
    
    # 6. Value-Based Optimization (Bottom-Middle)
    draw_box(55, 20, 18, 12, "Value-Based Filter", "EIP > 0 & Uplift >= 0", color_decision)
    
    # 7. Deliverables & Dashboard (Bottom-Left)
    draw_box(30, 20, 18, 12, "Campaign Optimizer", "Targeting List & Streamlit App", color_output)
    
    # Draw Pipeline Connections (Flow arrows)
    draw_arrow(23, 64, 30, 64, "Data Load")
    draw_arrow(48, 64, 55, 64, "Compute features")
    draw_arrow(73, 64, 80, 64, "Train models")
    
    # Downward arrow to scoring
    draw_arrow(87, 58, 87, 32, "Evaluate All Users")
    
    draw_arrow(80, 26, 73, 26, "Expected Profit (EIP)")
    draw_arrow(55, 26, 48, 26, "Generate Output")
    
    # Loop back to DB for closed loop evaluation
    draw_arrow(39, 20, 39, 10, "Targeting List")
    draw_box(30, 2, 18, 8, "SMS / Email Gateway", "Execute Campaign", color_db)
    
    draw_arrow(48, 6, 64, 6, "New Transactions")
    draw_cylinder(64, 2, 18, 8, "Control Feedback", "Measure Actual Uplift", color_db)
    draw_arrow(73, 10, 14, 58, "Continuous Model Update")

    # Legend / Info card
    legend_box = patches.Rectangle(
        (2, 2), 22, 18,
        facecolor='#1e2530', edgecolor='#2d3748', linewidth=1, zorder=2
    )
    ax.add_patch(legend_box)
    ax.text(3, 18, "Legend / Colors:", color='#ffffff', fontsize=9, fontweight='bold', zorder=3)
    
    # Draw legend icons and text
    categories = [
        ("Data & Inputs", color_db),
        ("Process / ETL", color_process),
        ("Machine Learning", color_model),
        ("Value Optimization", color_decision),
        ("Actionable Output", color_output)
    ]
    for i, (name, col) in enumerate(categories):
        rect = patches.Rectangle((3, 14.5 - i*2.7), 2, 1.8, facecolor=col, zorder=3)
        ax.add_patch(rect)
        ax.text(6, 15.3 - i*2.7, name, color='#cbd5e0', fontsize=8, zorder=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='#0f1115')
    plt.close()
    print(f"Workflow diagram generated and saved to '{output_path}'")

if __name__ == '__main__':
    draw_diagram()
