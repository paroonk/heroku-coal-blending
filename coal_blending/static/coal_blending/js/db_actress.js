$(document).ready(function() {
    $('#dbTable').DataTable({
        paging: false,
    });

    $(document).on("click", '[data-toggle="lightbox"]', function(event) {
        event.preventDefault();
        $(this).ekkoLightbox();
    });
});