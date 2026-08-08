from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from typing import Any, Dict, List, Optional
import os
import base64
from app.utils.logger import get_logger

logger = get_logger(__name__)

class ReportGenerator:
    """
    Service creating professional reports: vector PDF documents using ReportLab
    and highly styled standalone printable HTML files.
    """

    @staticmethod
    def generate_pdf_report(
        dataset_info: Dict[str, Any],
        algorithm_name: str,
        metrics: Dict[str, Any],
        rules: List[str],
        cm_image_path: Optional[str],
        roc_image_path: Optional[str],
        tree_image_path: Optional[str],
        output_path: str
    ):
        """
        Compile validation outcomes and visualization plots into a structured PDF document.
        """
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
        )
        story = []
        
        styles = getSampleStyleSheet()
        
        # Define premium custom styles matching project theme
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=colors.HexColor('#6366f1'),
            spaceAfter=15
        )
        
        h1_style = ParagraphStyle(
            'SectionH1',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=18,
            textColor=colors.HexColor('#1e1b4b'),
            spaceBefore=12,
            spaceAfter=8,
            keepWithNext=True
        )
        
        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#334155')
        )
        
        mono_style = ParagraphStyle(
            'Mono',
            parent=styles['Code'],
            fontName='Courier',
            fontSize=9,
            leading=11,
            textColor=colors.HexColor('#0f172a')
        )
        
        # 1. Header
        story.append(Paragraph("DataMine AI Classifier", title_style))
        story.append(Paragraph(f"<b>Algorithm Performance Report:</b> {algorithm_name}", body_style))
        story.append(Spacer(1, 15))
        
        # 2. Section: Dataset info
        story.append(Paragraph("1. Dataset Specifications", h1_style))
        meta_data = [
            [Paragraph("<b>Relation Name:</b>", body_style), Paragraph(str(dataset_info.get("relation_name", "N/A")), body_style)],
            [Paragraph("<b>Total Instances:</b>", body_style), Paragraph(str(dataset_info.get("num_instances", "N/A")), body_style)],
            [Paragraph("<b>Attributes Count:</b>", body_style), Paragraph(str(dataset_info.get("num_attributes", "N/A")), body_style)],
            [Paragraph("<b>Target Class Variable:</b>", body_style), Paragraph(str(dataset_info.get("class_attribute", "N/A")), body_style)]
        ]
        meta_table = Table(meta_data, colWidths=[150, 350])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('PADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 15))
        
        # 3. Section: Performance Metrics
        story.append(Paragraph("2. Model Evaluation Outcomes", h1_style))
        metrics_data = [
            [Paragraph("<b>Metric Name</b>", body_style), Paragraph("<b>Value</b>", body_style)],
            [Paragraph("Accuracy score", body_style), Paragraph(f"{metrics.get('accuracy', 0.0)*100:.2f}%", body_style)],
            [Paragraph("Precision (weighted)", body_style), Paragraph(f"{metrics.get('precision', 0.0)*100:.2f}%", body_style)],
            [Paragraph("Recall (weighted)", body_style), Paragraph(f"{metrics.get('recall', 0.0)*100:.2f}%", body_style)],
            [Paragraph("F1-Score (weighted)", body_style), Paragraph(f"{metrics.get('f1_score', 0.0)*100:.2f}%", body_style)],
            [Paragraph("Cohen's Kappa statistic", body_style), Paragraph(f"{metrics.get('cohen_kappa', 0.0):.4f}", body_style)],
            [Paragraph("ROC Area Under Curve (AUC)", body_style), Paragraph(f"{metrics.get('roc_auc', 0.0):.4f}" if metrics.get('roc_auc') is not None else "N/A", body_style)],
            [Paragraph("Memory overhead delta", body_style), Paragraph(f"{metrics.get('memory_used_mb', 0.0):.4f} MB", body_style)],
            [Paragraph("Training duration speed", body_style), Paragraph(f"{metrics.get('execution_time_ms', 0.0)} ms", body_style)]
        ]
        metrics_table = Table(metrics_data, colWidths=[250, 250])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e8f0')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('PADDING', (0,0), (-1,-1), 6),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 15))
        
        # 4. Section: Visual plots (Confusion Matrix, ROC)
        plots = []
        if cm_image_path and os.path.exists(cm_image_path):
            plots.append(Image(cm_image_path, width=220, height=180))
        if roc_image_path and os.path.exists(roc_image_path):
            plots.append(Image(roc_image_path, width=220, height=180))
            
        if plots:
            story.append(Paragraph("3. Visual Analytics Charts", h1_style))
            # Wrap plots inside a table side-by-side
            plots_table = Table([plots], colWidths=[250, 250])
            plots_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            story.append(KeepTogether(plots_table))
            story.append(Spacer(1, 15))
            
        # 5. Section: Rules (if decision tree algorithm)
        if rules:
            story.append(Paragraph("4. Extracted Logic Rules pathways", h1_style))
            rules_blocks = []
            for r in rules[:15]: # Show at most 15 rules in the PDF
                rules_blocks.append(Paragraph(r.replace('\n', '<br/>'), mono_style))
                rules_blocks.append(Spacer(1, 8))
            story.append(KeepTogether(rules_blocks))
            
        # 6. Tree PNG Image
        if tree_image_path and os.path.exists(tree_image_path):
            story.append(Spacer(1, 15))
            story.append(Paragraph("5. Compiled Decision Tree Graphviz representation", h1_style))
            story.append(KeepTogether(Image(tree_image_path, width=480, height=300)))
            
        doc.build(story)

    @staticmethod
    def generate_html_report(
        dataset_info: Dict[str, Any],
        algorithm_name: str,
        metrics: Dict[str, Any],
        rules: List[str],
        cm_image_path: Optional[str],
        roc_image_path: Optional[str],
        tree_image_path: Optional[str],
        output_path: str
    ):
        """
        Generate standalone printable HTML report incorporating inline embedded images.
        """
        
        def to_base64_img(img_path: Optional[str]) -> str:
            if img_path and os.path.exists(img_path):
                try:
                    with open(img_path, "rb") as f:
                        encoded = base64.b64encode(f.read()).decode("utf-8")
                        return f"data:image/png;base64,{encoded}"
                except Exception:
                    pass
            return ""
            
        cm_base64 = to_base64_img(cm_image_path)
        roc_base64 = to_base64_img(roc_image_path)
        tree_base64 = to_base64_img(tree_image_path)
        
        rules_html = ""
        for r in rules:
            rules_html += f'<div class="rule-box"><pre>{r}</pre></div>'
            
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>DataMine AI Classifier Report - {algorithm_name}</title>
  <style>
    body {{
      font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
      color: #334155;
      background-color: #ffffff;
      margin: 0;
      padding: 2.5rem;
      line-height: 1.5;
    }}
    .header {{
      border-bottom: 2px solid #6366f1;
      padding-bottom: 1.5rem;
      margin-bottom: 2rem;
    }}
    .title {{
      color: #6366f1;
      font-size: 2.2rem;
      font-weight: 800;
      margin: 0 0 0.5rem 0;
    }}
    .subtitle {{
      font-size: 1.1rem;
      color: #475569;
      margin: 0;
    }}
    h2 {{
      color: #1e1b4b;
      font-size: 1.4rem;
      border-bottom: 1px solid #e2e8f0;
      padding-bottom: 0.5rem;
      margin-top: 2rem;
      margin-bottom: 1rem;
      page-break-after: avoid;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 1.5rem;
    }}
    th, td {{
      padding: 0.75rem 1rem;
      border: 1px solid #cbd5e1;
      text-align: left;
    }}
    th {{
      background-color: #f1f5f9;
      font-weight: 600;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
    }}
    .img-box {{
      border: 1px solid #cbd5e1;
      padding: 0.5rem;
      text-align: center;
      background-color: #f8fafc;
    }}
    .img-box img {{
      max-width: 100%;
      height: auto;
    }}
    .rules-container {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 0.75rem;
    }}
    .rule-box {{
      background-color: #f8fafc;
      border: 1px solid #e2e8f0;
      padding: 0.75rem 1.25rem;
      border-radius: 4px;
    }}
    pre {{
      margin: 0;
      font-family: monospace;
      font-size: 0.9rem;
      white-space: pre-wrap;
    }}
    @media print {{
      body {{
        padding: 0;
      }}
      .no-print {{
        display: none;
      }}
      h2 {{
        page-break-inside: avoid;
      }}
      .img-box {{
        page-break-inside: avoid;
      }}
      .rule-box {{
        page-break-inside: avoid;
      }}
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1 class="title">DataMine AI Classifier</h1>
    <p class="subtitle">Performance benchmark report: <strong>{algorithm_name}</strong></p>
  </div>
  
  <div class="no-print" style="margin-bottom: 1.5rem; text-align: right;">
    <button onclick="window.print()" style="padding: 0.75rem 1.5rem; font-size: 1rem; font-weight: 600; background-color: #6366f1; color: white; border: none; border-radius: 4px; cursor: pointer;">
      Print Report
    </button>
  </div>

  <h2>1. Dataset Specifications</h2>
  <table>
    <tr>
      <th style="width: 250px;">Relation Name</th>
      <td>{dataset_info.get("relation_name", "N/A")}</td>
    </tr>
    <tr>
      <th>Total Instances</th>
      <td>{dataset_info.get("num_instances", "N/A")}</td>
    </tr>
    <tr>
      <th>Attributes count</th>
      <td>{dataset_info.get("num_attributes", "N/A")}</td>
    </tr>
    <tr>
      <th>Target Class Variable</th>
      <td>{dataset_info.get("class_attribute", "N/A")}</td>
    </tr>
  </table>

  <h2>2. Model Evaluation Outcomes</h2>
  <table>
    <thead>
      <tr>
        <th>Metric Name</th>
        <th>Value</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Accuracy score</td>
        <td><strong>{metrics.get('accuracy', 0.0)*100:.2f}%</strong></td>
      </tr>
      <tr>
        <td>Precision (weighted)</td>
        <td>{metrics.get('precision', 0.0)*100:.2f}%</td>
      </tr>
      <tr>
        <td>Recall (weighted)</td>
        <td>{metrics.get('recall', 0.0)*100:.2f}%</td>
      </tr>
      <tr>
        <td>F1-Score (weighted)</td>
        <td>{metrics.get('f1_score', 0.0)*100:.2f}%</td>
      </tr>
      <tr>
        <td>Cohen's Kappa statistic</td>
        <td>{metrics.get('cohen_kappa', 0.0):.4f}</td>
      </tr>
      <tr>
        <td>ROC Area Under Curve (AUC)</td>
        <td>{f"{metrics.get('roc_auc', 0.0):.4f}" if metrics.get('roc_auc') is not None else "N/A"}</td>
      </tr>
      <tr>
        <td>Memory overhead delta</td>
        <td>{metrics.get('memory_used_mb', 0.0):.4f} MB</td>
      </tr>
      <tr>
        <td>Training duration speed</td>
        <td>{metrics.get('execution_time_ms', 0.0)} ms</td>
      </tr>
    </tbody>
  </table>

  <h2>3. Visual Analytics Charts</h2>
  <div class="grid-2">
    {f'<div class="img-box"><h3>Confusion Matrix Heatmap</h3><img src="{cm_base64}" alt="Confusion Matrix"/></div>' if cm_base64 else ''}
    {f'<div class="img-box"><h3>ROC Curve Chart</h3><img src="{roc_base64}" alt="ROC Curve"/></div>' if roc_base64 else ''}
  </div>

  {f'<h2>4. Extracted Logic Rules Pathways</h2><div class="rules-container">{rules_html}</div>' if rules else ''}

  {f'<h2>5. Decision Tree Graphic Representation</h2><div class="img-box" style="margin-top: 1.5rem;"><img src="{tree_base64}" alt="Decision Tree"/></div>' if tree_base64 else ''}
</body>
</html>
"""
        with open(output_path, "w") as f:
            f.write(html_content)
