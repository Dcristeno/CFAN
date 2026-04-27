import logging

from utils.metrics import Evaluator


def do_inference(model, test_img_loader, test_txt_loader):
    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")
    evaluator = Evaluator(test_img_loader, test_txt_loader)
    return evaluator.eval(model.eval(), return_details=True)
