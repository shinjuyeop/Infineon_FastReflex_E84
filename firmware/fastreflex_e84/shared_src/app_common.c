/******************************************************************************
* File Name:   app_common.c
*
* Description: This file contains the implementation of common functions used
*              by the application.
*
* Related Document: See README.md
*
*
*******************************************************************************
* (c) 2025-2026, Infineon Technologies AG, or an affiliate of Infineon
* Technologies AG. All rights reserved.
* This software, associated documentation and materials ("Software") is
* owned by Infineon Technologies AG or one of its affiliates ("Infineon")
* and is protected by and subject to worldwide patent protection, worldwide
* copyright laws, and international treaty provisions. Therefore, you may use
* this Software only as provided in the license agreement accompanying the
* software package from which you obtained this Software. If no license
* agreement applies, then any use, reproduction, modification, translation, or
* compilation of this Software is prohibited without the express written
* permission of Infineon.
*
* Disclaimer: UNLESS OTHERWISE EXPRESSLY AGREED WITH INFINEON, THIS SOFTWARE
* IS PROVIDED AS-IS, WITH NO WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
* INCLUDING, BUT NOT LIMITED TO, ALL WARRANTIES OF NON-INFRINGEMENT OF
* THIRD-PARTY RIGHTS AND IMPLIED WARRANTIES SUCH AS WARRANTIES OF FITNESS FOR A
* SPECIFIC USE/PURPOSE OR MERCHANTABILITY.
* Infineon reserves the right to make changes to the Software without notice.
* You are responsible for properly designing, programming, and testing the
* functionality and safety of your intended application of the Software, as
* well as complying with any legal requirements related to its use. Infineon
* does not guarantee that the Software will be free from intrusion, data theft
* or loss, or other breaches ("Security Breaches"), and Infineon shall have
* no liability arising out of any Security Breaches. Unless otherwise
* explicitly approved by Infineon, the Software may not be used in any
* application where a failure of the Product or any consequences of the use
* thereof can reasonably be expected to result in personal injury.
*******************************************************************************/
#include "app_common.h"

#include "cybsp.h"
#include "cy_pdl.h"

#include "cy_retarget_io.h"

/*******************************************************************************
* Global Variables
*******************************************************************************/
/* For the RetargetIO (Debug UART) usage */
cy_stc_scb_uart_context_t  CYBSP_DEBUG_UART_context;  /** UART context */
mtb_hal_uart_t mtb_ml_retarget_io_uart_obj;  /** UART HAL object */

/*******************************************************************************
* Function Name: app_retarget_io_init
********************************************************************************
* Summary:
* User defined function to initialize the debug UART.
*
* Parameters:
*  baudrate: baudrate to be applied to the UART
*
* Return:
*  void
*
*******************************************************************************/
void app_retarget_io_init(uint32_t baudrate)
{
    cy_rslt_t result;

    /* Initialize the SCB UART */
    result = (cy_rslt_t)Cy_SCB_UART_Init(CYBSP_DEBUG_UART_HW,
                                        &CYBSP_DEBUG_UART_config,
                                        &CYBSP_DEBUG_UART_context);

    /* UART init failed. Stop program execution */
    if (CY_RSLT_SUCCESS != result)
    {
        handle_error();
    }

    /* Enable the SCB UART */
    Cy_SCB_UART_Enable(CYBSP_DEBUG_UART_HW);

    result = mtb_hal_uart_setup(&mtb_ml_retarget_io_uart_obj,
                                &CYBSP_DEBUG_UART_hal_config,
                                &CYBSP_DEBUG_UART_context, NULL);

    /* UART setup failed. Stop program execution */
    if (CY_RSLT_SUCCESS != result)
    {
        handle_error();
    }

    /* Set the UART baudrate */
    result = mtb_hal_uart_set_baud(&mtb_ml_retarget_io_uart_obj, baudrate, NULL);

    /* UART setup failed. Stop program execution */
    if (CY_RSLT_SUCCESS != result)
    {
        handle_error();
    }

    /* Initialize retarget-io to use the debug UART port */
    result = cy_retarget_io_init(&mtb_ml_retarget_io_uart_obj);

    /* retarget-io init failed. Stop program execution */
    if (CY_RSLT_SUCCESS != result)
    {
        handle_error();
    }
}


/*******************************************************************************
* Function Name: handle_error
********************************************************************************
* Summary:
* User defined error handling function
*
* Parameters:
*  void
*
* Return:
*  void
*
*******************************************************************************/
void handle_error(void)
{
    /* Disable all interrupts. */
    __disable_irq();

    CY_ASSERT(0);
}

/* [] END OF FILE */