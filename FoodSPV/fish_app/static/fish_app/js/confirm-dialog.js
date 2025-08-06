function confirmDelete(message) {
    return new Promise((resolve) => {
        const dialog = document.createElement('div');
        dialog.className = 'modal fade';
        dialog.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Confirm Delete</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <p>${message}</p>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-primary" id="confirmBtn">Confirm</button>
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(dialog);
        
        const modal = new bootstrap.Modal(dialog);
        modal.show();
        
        dialog.querySelector('#confirmBtn').onclick = () => {
            modal.hide();
            resolve(true);
        };
        
        dialog.addEventListener('hidden.bs.modal', () => {
            document.body.removeChild(dialog);
            resolve(false);
        });
    });
} 