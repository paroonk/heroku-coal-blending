$(document).ready(function() {
    $('#dbTable').DataTable({
        'lengthMenu': [[10, 25, 50, -1], [10, 25, 50, "All"]]
    });

    $(document).on("click", '[data-toggle="lightbox"]', function(event) {
        event.preventDefault();
        $(this).ekkoLightbox();
    });
});