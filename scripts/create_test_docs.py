"""
Generate Realistic Multi-Page Business Test Documents for OCR and RAG Evaluation.

Creates:
1. samples/master_service_agreement.pdf (4 pages: Scope, SLAs, Fees, Liability/Termination)
2. samples/cloud_services_invoice.pdf (2 pages: Itemized billing, tax ID, payment terms)
3. samples/non_disclosure_agreement.pdf (2 pages: Confidentiality, governing law)
"""

import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def create_master_service_agreement(output_path: str):
    """Generate 4-page realistic enterprise Master Service Agreement (MSA)."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1E293B"),
        alignment=1,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2563EB"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
    )

    story = []

    # ------------------ PAGE 1 ------------------
    story.append(Paragraph("ACME ENTERPRISE SOLUTIONS", title_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("MASTER SERVICES AGREEMENT (MSA)", title_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(
        Paragraph(
            "This Master Services Agreement ('Agreement') is entered into as of October 1, 2026 ('Effective Date'), "
            "by and between <b>Acme Cloud Solutions Inc.</b>, a Delaware corporation having its principal office at "
            "100 Innovation Way, Suite 400, Austin, TX 78701 ('Provider'), and <b>Apex Global Logistics LLC</b>, "
            "a California limited liability company ('Customer').",
            body_style,
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("1. PURPOSE AND SCOPE OF SERVICES", heading_style))
    story.append(
        Paragraph(
            "Provider agrees to deliver intelligent document processing, cloud data ingestion, and retrieval-augmented "
            "generation infrastructure as described in Statement of Work #101 (SOW-101). Customer shall have a non-exclusive, "
            "worldwide license to access Provider's proprietary serverless analytics platforms during the term of this Agreement.",
            body_style,
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("2. TERM AND RENEWAL", heading_style))
    story.append(
        Paragraph(
            "The initial term of this Agreement shall commence on the Effective Date and continue for a period of twenty-four (24) months. "
            "Thereafter, this Agreement shall automatically renew for successive twelve (12) month periods unless either party provides written "
            "notice of non-renewal at least sixty (60) days prior to the expiration of the then-current term.",
            body_style,
        )
    )
    story.append(PageBreak())

    # ------------------ PAGE 2 ------------------
    story.append(Paragraph("SERVICE LEVEL AGREEMENT & UPTIME GUARANTEES", title_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("3. SERVICE LEVEL COMMITMENTS (SLA)", heading_style))
    story.append(
        Paragraph(
            "Provider guarantees that the Intelligent Document Processing Pipeline and Query API endpoints will maintain "
            "a Monthly Uptime Percentage of at least <b>99.9%</b> during each billing cycle, excluding Scheduled Maintenance.",
            body_style,
        )
    )
    story.append(Spacer(1, 0.1 * inch))

    sla_data = [
        ["Monthly Uptime Percentage", "Service Credit Percentage", "Response Time Target"],
        [">= 99.9%", "0% (Normal Operations)", "< 200 ms latency"],
        ["99.0% to 99.89%", "10% of monthly fee", "< 500 ms latency"],
        ["95.0% to 98.99%", "25% of monthly fee", "< 1,200 ms latency"],
        ["< 95.0%", "50% of monthly fee", "Immediate escalation"],
    ]
    sla_table = Table(sla_data, colWidths=[2.2 * inch, 2.2 * inch, 2.2 * inch])
    sla_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ])
    )
    story.append(sla_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("4. SCHEDULED MAINTENANCE WINDOWS", heading_style))
    story.append(
        Paragraph(
            "Provider may perform routine maintenance during designated maintenance windows on Sundays between "
            "02:00 UTC and 06:00 UTC. Provider will notify Customer at least forty-eight (48) hours in advance of any maintenance "
            "expected to exceed thirty (30) minutes.",
            body_style,
        )
    )
    story.append(PageBreak())

    # ------------------ PAGE 3 ------------------
    story.append(Paragraph("PRICING, FEES & PAYMENT SCHEDULE", title_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("5. FEES AND INVOICING", heading_style))
    story.append(
        Paragraph(
            "Customer shall pay a recurring base platform fee of <b>$25,000 USD per month</b> for up to 500,000 document pages processed. "
            "Excess document pages will be billed at an overage rate of <b>$0.015 per page</b>. All fees are exclusive of applicable taxes.",
            body_style,
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("6. PAYMENT TERMS AND LATE FEES", heading_style))
    story.append(
        Paragraph(
            "Invoices are issued on the first business day of each calendar month. Payment terms are strictly <b>Net 30 days</b> "
            "from the date of invoice. Late payments shall accrue interest at the rate of <b>1.5% per month</b> (or the maximum permitted "
            "by law, whichever is lower) on all past-due balances. In the event of non-payment exceeding 45 days, Provider reserves the right "
            "to suspend API access upon five (5) days written notice.",
            body_style,
        )
    )
    story.append(PageBreak())

    # ------------------ PAGE 4 ------------------
    story.append(Paragraph("LIABILITY, TERMINATION & SIGNATURES", title_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("7. LIMITATION OF LIABILITY", heading_style))
    story.append(
        Paragraph(
            "TO THE MAXIMUM EXTENT PERMITTED BY LAW, IN NO EVENT SHALL EITHER PARTY'S AGGREGATE LIABILITY ARISING OUT OF OR "
            "RELATED TO THIS AGREEMENT EXCEED <b>$500,000 USD (FIVE HUNDRED THOUSAND DOLLARS)</b>. NEITHER PARTY SHALL BE LIABLE FOR "
            "ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, INCLUDING LOSS OF PROFITS, DATA LOSS, OR BUSINESS "
            "INTERRUPTION, REGARDLESS OF THE THEORY OF LIABILITY.",
            body_style,
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("8. TERMINATION FOR CAUSE AND CONVENIENCE", heading_style))
    story.append(
        Paragraph(
            "Either party may terminate this Agreement for convenience upon providing at least <b>thirty (30) days prior written notice</b> "
            "to the other party. Either party may terminate immediately upon written notice if the other party materially breaches this "
            "Agreement and fails to cure such breach within fifteen (15) days of receiving notice.",
            body_style,
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("9. GOVERNING LAW AND VENUE", heading_style))
    story.append(
        Paragraph(
            "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, "
            "without regard to its conflict of law principles. Any dispute shall be resolved exclusively in the state or federal courts "
            "located in New Castle County, Delaware.",
            body_style,
        )
    )
    story.append(Spacer(1, 0.3 * inch))

    sig_data = [
        ["For Acme Cloud Solutions Inc.:", "For Apex Global Logistics LLC:"],
        ["Signature: __________________________", "Signature: __________________________"],
        ["Name: Sarah Jenkins", "Name: Marcus Vance"],
        ["Title: Chief Operating Officer", "Title: VP of Technology"],
        ["Date: October 1, 2026", "Date: October 1, 2026"],
    ]
    sig_table = Table(sig_data, colWidths=[3.3 * inch, 3.3 * inch])
    sig_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(sig_table)

    doc.build(story)
    print(f"Generated: {output_path}")


def create_cloud_services_invoice(output_path: str):
    """Generate 2-page detailed cloud services invoice."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "InvTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
    )
    body_style = ParagraphStyle(
        "InvBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
    )

    story = []

    # ------------------ PAGE 1 ------------------
    story.append(Paragraph("INVOICE #INV-2026-8891", title_style))
    story.append(Spacer(1, 0.15 * inch))

    info_data = [
        ["Issuer:", "Bill To:"],
        ["Acme Cloud Solutions Inc.\n100 Innovation Way, Suite 400\nAustin, TX 78701\nTax ID: US-948291034",
         "Apex Global Logistics LLC\n450 Fremont Street, 12th Floor\nSan Francisco, CA 94105\nAttn: Accounts Payable"],
        ["Invoice Date: October 15, 2026", "Payment Terms: Net 30 Days"],
        ["Due Date: November 14, 2026", "Currency: USD ($)"],
    ]
    info_table = Table(info_data, colWidths=[3.3 * inch, 3.3 * inch])
    info_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.append(info_table)
    story.append(Spacer(1, 0.25 * inch))

    items_data = [
        ["Line", "Description", "Usage / Qty", "Unit Price", "Total Amount"],
        ["1", "IDP OCR Processing Engine (Amazon Textract tier)", "420,000 pages", "$0.015", "$6,300.00"],
        ["2", "Serverless FAISS Vector Storage & Search Indexing", "1,200 hours", "$0.85", "$1,020.00"],
        ["3", "Open-Source LLM Serverless Inference (Llama 3.2)", "85,000 queries", "$0.004", "$340.00"],
        ["4", "Enterprise Cloud Support & 99.9% SLA Assurance", "1 month", "$1,500.00", "$1,500.00"],
        ["", "", "", "Subtotal:", "$9,160.00"],
        ["", "", "", "Applicable Sales Tax (8.25%):", "$755.70"],
        ["", "", "", "Total Due:", "$9,915.70"],
    ]
    items_table = Table(items_data, colWidths=[0.5 * inch, 2.7 * inch, 1.2 * inch, 1.0 * inch, 1.2 * inch])
    items_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
            ("GRID", (0, 0), (-1, 4), 0.5, colors.HexColor("#CBD5E1")),
            ("FONTNAME", (3, 4), (-1, -1), "Helvetica-Bold"),
            ("BACKGROUND", (3, 6), (4, 6), colors.HexColor("#FEF9C3")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.append(items_table)
    story.append(PageBreak())

    # ------------------ PAGE 2 ------------------
    story.append(Paragraph("PAYMENT REMITTANCE & BANK INSTRUCTIONS", title_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(
        Paragraph(
            "Please remit electronic payments via Wire Transfer or Automated Clearing House (ACH) using the banking details below. "
            "Please include Invoice Number <b>INV-2026-8891</b> on all payment remittances.",
            body_style,
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    bank_data = [
        ["Beneficiary Bank:", "Silicon Valley Commerce Bank, N.A."],
        ["Bank Address:", "3003 Tasman Drive, Santa Clara, CA 95054"],
        ["Routing Number (ABA):", "121140399"],
        ["Swift / BIC Code:", "SVBKUS6S"],
        ["Account Name:", "Acme Cloud Solutions Inc. - Operating Account"],
        ["Account Number:", "8839-4401-2910"],
    ]
    bank_table = Table(bank_data, colWidths=[2.5 * inch, 4.0 * inch])
    bank_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(bank_table)
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("Billing Contact Information:", styles["Heading3"]))
    story.append(
        Paragraph(
            "For inquiries regarding this invoice, please reach out to our billing operations team at "
            "<b>billing@acmecloudsolutions.com</b> or by phone at <b>+1 (800) 555-0199</b> (Monday through Friday, 9am - 5pm CT).",
            body_style,
        )
    )

    doc.build(story)
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    os.makedirs("/Users/wilson/aws-idp-rag/samples", exist_ok=True)
    create_master_service_agreement("/Users/wilson/aws-idp-rag/samples/master_service_agreement.pdf")
    create_cloud_services_invoice("/Users/wilson/aws-idp-rag/samples/cloud_services_invoice.pdf")
    print("All sample business documents created successfully!")
