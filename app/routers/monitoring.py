"""Model monitoring routes."""

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from app.dependencies import get_monitoring_service
from app.services.monitoring_service import MonitoringService

router = APIRouter()

@router.post("/monitor/report")
async def monitor_report(request: Request, reference: UploadFile = File(...), current: UploadFile = File(...), alpha: float = Form(0.05), service: MonitoringService = Depends(get_monitoring_service)):
    report = service.create_report(await reference.read(), await current.read(), alpha, request.state.request_id)
    return JSONResponse(content=report)
